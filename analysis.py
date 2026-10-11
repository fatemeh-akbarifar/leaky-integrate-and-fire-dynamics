"""Regenerate every reported table and figure: python analysis.py."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib, json, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Parameters, simulate, analytic_rate, measured_rate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7fafc','axes.facecolor':'#f7fafc','savefig.facecolor':'#f7fafc'})
    rows=[]
    for ref in (1.,2.,5.):
        p=replace(Parameters(),refractory_ms=ref)
        for current in np.linspace(0,.5,21):
            t,v,s=simulate(float(current),p)
            fine=measured_rate(simulate(float(current),p,dt_ms=.05)[2])
            rate=measured_rate(s)
            assert abs(rate-fine) <= max(2.,.02*fine)
            rows.append(dict(current_na=current,**asdict(p),dt_ms=.1,duration_ms=1000.,window_start_ms=100.,window_end_ms=1000.,spike_count=int(((s>100)&(s<=1000)).sum()),rate_hz=rate,analytic_hz=analytic_rate(current,p),fine_rate_hz=fine))
    df=pd.DataFrame(rows);df.to_csv(OUT/'firing-rates.csv',index=False,float_format='%.8g')
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,sharey=True)
    trace=[]
    for ax,current,color in zip(axes,(.1,.2,.4),('#385a7c','#168a88','#c05a44')):
        t,v,s=simulate(current)
        ax.plot(t,v,color=color,lw=1.4)
        shown=s[s<=200]
        ax.scatter(shown,np.full(len(shown),-50.),marker='|',s=100,color=color,zorder=4)
        ax.axhline(-50,color='#64748b',ls='--',lw=.8)
        ax.set(ylabel='Voltage (mV)',xlim=(0,200),ylim=(-66,-48),title=f'{current:.1f} nA input')
        trace.extend(dict(current_na=current,time_ms=a,voltage_mv=b) for a,b in zip(t[t<=200],v[t<=200]))
    axes[-1].set_xlabel('Time (ms)');fig.suptitle('A threshold turns gradual charging into discrete events',fontsize=16)
    fig.tight_layout();fig.savefig(OUT/'voltage-traces.png',dpi=170);plt.close(fig)
    pd.DataFrame(trace).to_csv(OUT/'voltage-traces.csv',index=False,float_format='%.8g')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    baseline=df[df.refractory_ms==2]
    axes[0].plot(baseline.current_na,baseline.rate_hz,'o',color='#168a88',label='Euler simulation; 900 ms count')
    dense=np.linspace(0,.5,500)
    axes[0].plot(dense,[analytic_rate(i) for i in dense],color='#385a7c',label='Continuous-time prediction')
    for ref,color in zip((1,2,5),('#168a88','#385a7c','#c05a44')):
        d=df[df.refractory_ms==ref];axes[1].plot(d.current_na,d.rate_hz,'o-',ms=4,color=color,label=f'{ref} ms recovery')
    for ax in axes:
        ax.axvline(.15,color='#94a3b8',ls=':',label='Threshold current: 0.15 nA')
        ax.set(xlabel='Input current (nA)',ylabel='Firing rate (Hz)',ylim=(0,140));ax.legend(fontsize=8,frameon=False)
    axes[0].set_title('Can the simulation recover the equation?');axes[1].set_title('Recovery time limits repeated firing')
    fig.tight_layout();fig.savefig(OUT/'firing-curves.png',dpi=170);plt.close(fig)
    checks={'conditions':len(df),'maximum_step_refinement_difference_hz':float(abs(df.rate_hz-df.fine_rate_hz).max()),'maximum_analytic_difference_hz':float(abs(df.rate_hz-df.analytic_hz).max()),'all_refinement_checks_passed':True}
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    manifest={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'data':'Deterministic simulation; no external dataset','sources_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('analysis.py','model.py')},'measurement_window':'(100, 1000] ms','units':'mV, ms, nA, megaohms'}
    (OUT/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(checks));print(df[np.isclose(df.current_na,.4)][['refractory_ms','rate_hz','analytic_hz']].to_string(index=False))
    return df

if __name__=='__main__':main()
