from pathlib import Path
import glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
BASE=Path('/mnt/data')
OUT=BASE/'runoff_project_final'
DATA=OUT/'data'; RES=OUT/'results'; FIG=OUT/'figures'; CODE=OUT/'code'
for p in [DATA,RES,FIG,CODE]: p.mkdir(parents=True, exist_ok=True)
STATIONS=[
 {'name':'station_09520500_20380357','usgs_file':'09520500_Strt_2021-04-20_EndAt_2023-04-21.csv','usgs_id':'09520500','stream_id':'20380357'},
 {'name':'station_11266500_21609641','usgs_file':'11266500_Strt_2021-04-20_EndAt_2023-04-21.csv','usgs_id':'11266500','stream_id':'21609641'},]

def read_usgs(path):
 df=pd.read_csv(path)
 df['DateTime']=pd.to_datetime(df['DateTime'], utc=True, errors='coerce')
 df['USGSFlowValue']=pd.to_numeric(df['USGSFlowValue'], errors='coerce')
 q='00060_cd' if '00060_cd' in df.columns else None
 df['estimated_flag_15min']=df[q].astype(str).str.contains('e',case=False,na=False).astype(int) if q else 0
 hourly=df.dropna(subset=['DateTime']).set_index('DateTime').resample('1h').agg({'USGSFlowValue':'mean','estimated_flag_15min':'max'}).rename(columns={'USGSFlowValue':'usgs_flow','estimated_flag_15min':'estimated_flag'}).reset_index()
 return hourly

def read_nwm(sid):
 files=sorted(glob.glob(str(BASE/f'streamflow_{sid}_*.csv')))
 arr=[]
 for f in files:
  d=pd.read_csv(f); d['source_file']=Path(f).name; arr.append(d)
 df=pd.concat(arr,ignore_index=True)
 df['model_initialization_time']=pd.to_datetime(df['model_initialization_time'].astype(str).str.replace('_',' ',regex=False),format='%Y-%m-%d %H:%M:%S',utc=True,errors='coerce')
 df['model_output_valid_time']=pd.to_datetime(df['model_output_valid_time'].astype(str).str.replace('_',' ',regex=False),format='%Y-%m-%d %H:%M:%S',utc=True,errors='coerce')
 df['streamflow_value']=pd.to_numeric(df['streamflow_value'],errors='coerce')
 df=df.dropna(subset=['model_initialization_time','model_output_valid_time'])
 df['horizon']=((df['model_output_valid_time']-df['model_initialization_time']).dt.total_seconds()/3600).round().astype(int)
 df=df[df['horizon'].between(1,18)]
 raw=len(df)
 conflicts=df.groupby(['model_initialization_time','model_output_valid_time','horizon'])['streamflow_value'].nunique(dropna=False).gt(1).sum()
 df=df.sort_values(['model_initialization_time','model_output_valid_time','source_file']).drop_duplicates(['model_initialization_time','model_output_valid_time','horizon'])
 wide=df.pivot(index='model_initialization_time',columns='horizon',values='streamflow_value')
 wide.columns=[f'nwm_h{int(c)}' for c in wide.columns]
 wide=wide.reset_index().rename(columns={'model_initialization_time':'timestamp'})
 meta={'stream_id':sid,'files':len(files),'raw_rows':raw,'clean_rows':len(df),'duplicates_removed':raw-len(df),'conflicting_duplicates':int(conflicts),'missing_nwm_values':int(df['streamflow_value'].isna().sum()),'init_start':str(df['model_initialization_time'].min()),'init_end':str(df['model_initialization_time'].max()),'valid_start':str(df['model_output_valid_time'].min()),'valid_end':str(df['model_output_valid_time'].max())}
 return wide,meta

def station_table(st):
 usgs=read_usgs(BASE/st['usgs_file']); nwm,meta=read_nwm(st['stream_id'])
 idx=usgs.set_index('DateTime')['usgs_flow']
 df=nwm.merge(usgs.rename(columns={'DateTime':'timestamp'}),on='timestamp',how='left')
 df['station_name']=st['name']; df['usgs_id']=st['usgs_id']; df['stream_id']=st['stream_id']
 for h in range(1,19):
  df[f'obs_h{h}']=idx.reindex(df['timestamp']+pd.to_timedelta(h,unit='h')).to_numpy()
  df[f'resid_h{h}']=df[f'obs_h{h}']-df[f'nwm_h{h}']
  df[f'pct_error_h{h}']=(df[f'obs_h{h}']-df[f'nwm_h{h}'])/df[f'obs_h{h}'].replace(0,np.nan)*100
 df['resid_h1_lag1']=df['resid_h1'].shift(1)
 df['resid_h1_lag2']=df['resid_h1'].shift(2)
 df['resid_h1_lag3']=df['resid_h1'].shift(3)
 df['hour']=df['timestamp'].dt.hour; df['month']=df['timestamp'].dt.month; df['dayofyear']=df['timestamp'].dt.dayofyear
 df['hour_sin']=np.sin(2*np.pi*df['hour']/24); df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
 df['doy_sin']=np.sin(2*np.pi*df['dayofyear']/366); df['doy_cos']=np.cos(2*np.pi*df['dayofyear']/366)
 return df,meta

def metrics(obs,pred):
 obs=np.asarray(obs,dtype=float); pred=np.asarray(pred,dtype=float); m=np.isfinite(obs)&np.isfinite(pred); obs=obs[m]; pred=pred[m]
 if len(obs)<2: return dict(CC=np.nan,RMSE=np.nan,PBIAS=np.nan,NSE=np.nan)
 return dict(CC=float(np.corrcoef(obs,pred)[0,1]), RMSE=float(np.sqrt(np.mean((obs-pred)**2))), PBIAS=float((np.sum(obs)-np.sum(pred))/np.sum(obs)*100), NSE=float(1-np.sum((pred-obs)**2)/np.sum((obs-np.mean(obs))**2)))

tabs=[]; metas=[]
for st in STATIONS:
 df,meta=station_table(st); tabs.append(df); metas.append(meta); df.to_csv(DATA/f"{st['name']}_modeling_table.csv",index=False)
all_df=pd.concat(tabs,ignore_index=True).sort_values(['station_name','timestamp'])
all_df['split']=np.where(all_df['timestamp']<pd.Timestamp('2022-07-01',tz='UTC'),'train',np.where(all_df['timestamp']<pd.Timestamp('2022-10-01',tz='UTC'),'validation','test'))
all_df.to_csv(DATA/'combined_modeling_table.csv',index=False)
# complete rows usable for DL targets
req=[f'nwm_h{i}' for i in range(1,19)]+[f'obs_h{i}' for i in range(1,19)]+['usgs_flow','resid_h1_lag1','resid_h1_lag2','resid_h1_lag3']
complete=all_df.dropna(subset=req).copy(); complete.to_csv(DATA/'combined_modeling_table_complete.csv',index=False)
# Baseline NWM metrics only on test set
rows=[]
for station,sdf in complete[complete['split']=='test'].groupby('station_name'):
 for h in range(1,19):
  rows.append({'model':'Original NWM','station':station,'horizon':h,**metrics(sdf[f'obs_h{h}'],sdf[f'nwm_h{h}'])})
base=pd.DataFrame(rows); base.to_csv(RES/'original_nwm_test_metrics.csv',index=False)
base.groupby(['model','station'])[['CC','RMSE','PBIAS','NSE']].mean().reset_index().to_csv(RES/'original_nwm_test_summary.csv',index=False)
# plots
plt.figure(figsize=(8,5))
for station,sdf in base.groupby('station'):
 plt.plot(sdf['horizon'],sdf['RMSE'],marker='o',label=station)
plt.xlabel('Lead time (hours)'); plt.ylabel('RMSE'); plt.title('Original NWM Test RMSE by Horizon'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'original_nwm_rmse_by_horizon.png',dpi=200); plt.close()
plt.figure(figsize=(8,5))
for station,sdf in base.groupby('station'):
 plt.plot(sdf['horizon'],sdf['NSE'],marker='o',label=station)
plt.xlabel('Lead time (hours)'); plt.ylabel('NSE'); plt.title('Original NWM Test NSE by Horizon'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'original_nwm_nse_by_horizon.png',dpi=200); plt.close()
with open(RES/'data_summary.json','w') as f: json.dump({'station_metadata':metas,'rows':{'combined_raw':len(all_df),'combined_complete':len(complete),'train':int((complete.split=='train').sum()),'validation':int((complete.split=='validation').sum()),'test':int((complete.split=='test').sum())}},f,indent=2)
print('prepared', OUT)
print(pd.read_csv(RES/'original_nwm_test_summary.csv').to_string(index=False))
