import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

def mpPDF(var,q,pts):
    # Marcenko-Pastur pdf
#     q=T/N
    eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal=np.linspace(eMin,eMax,pts)
    pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf=pd.Series(pdf.squeeze(),index=eVal.squeeze())

    return pdf

def fitKDE(obs,bWidth=.25,kernel="gaussian",x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1:obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None:x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:x=x.reshape(-1,1)
    logProb=kde.score_samples(x) # log(density)
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf, kde


def errPDFs(var,eVal,q,bWidth,pts):
    # Fit error
    pdf0=mpPDF(var,q,pts) # theoretical pdf
    pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values)[0] # empirical pdf
    sse=np.sum((pdf1-pdf0)**2)
    return sse, pdf0, pdf1

def findMaxEval(eVal,q,bWidth,pts):
    # Find max random eVal by fitting Marcenko’s dist
    out=minimize(lambda *x:errPDFs(*x)[0],.5,args=(eVal,q,bWidth,pts),
    bounds=((1E-5,1-1E-5),))
    if out["success"]:
        var=out["x"][0]
    else:
        print('Marcenko-Pastur fit failed!')
        var=1
    eMax=var*(1+(1./q)**.5)**2
    sse, pdf0, pdf1 = errPDFs(var,eVal,q,bWidth,pts)
    return eMax,var, pdf0, pdf1