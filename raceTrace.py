import fastf1
import pandas as pd
import matplotlib.pyplot as plt


colors = {
    'ALB': '#005aff', 'SAI': '#012564', 'LEC': '#dc0000', 'HAD': '#2b4562',
    'DOO': '#ff117c', 'ALO': '#006f62', 'RUS': '#24ffff', 'OCO': "#6A6868",
    'STR': '#00413b', 'NOR': '#FF8700', 'HAM': '#ff2800', 'VER': '#23326A',
    'HUL': '#00e701', 'BEA': "#605e5e", 'PIA': '#FFD580', 'GAS': '#fe86bc',
    'LAW': "#50a8ac", 'ANT': "#a1fafa", 'TSU': '#356cac', 'BOR': '#008d01'
}


# load session data
session = fastf1.get_session(2025, 'Canada', 'R')
session.load()  

# prepare laps
laps = session.laps.loc[session.laps['LapTime'].notna()].copy()
laps.loc[:, 'LapTimeSec'] = laps['LapTime'].dt.total_seconds()
laps.loc[:, 'CumTime']    = laps.groupby('Driver')['LapTimeSec'].cumsum()
# build an absolute end‐of‐lap timestamp so we can match flags
laps.loc[:, 'LapEndDate'] = laps['LapStartDate'] + pd.to_timedelta(laps['LapTimeSec'], unit='s')

# identify P1 as reference driver
winner = session.results.loc[session.results['Position'] == 1, 'Abbreviation'].iloc[0]

# compute deltas
leader_ct = (laps[laps['Driver'] == winner]
             .set_index('LapNumber')['CumTime']
             .rename('LeaderCumTime'))
laps = laps.join(leader_ct, on='LapNumber')
laps.loc[:, 'GapToLeader'] = laps['CumTime'] - laps['LeaderCumTime']

# identify flagged laps
flag_laps = {}
for _, ev in session.race_control_messages.iterrows():
    msg = ev['Message'].lower()
    if 'red flag' in msg:
        typ = 'red'
    elif 'safety car' in msg or 'virtual safety car' in msg:
        typ = 'yellow'
    else:
        continue

    ts = ev['Time']  
    mask = (
        (laps['Driver'] == winner) &
        (laps['LapStartDate'] <= ts) &
        (laps['LapEndDate']   >= ts)
    )
    ref = laps.loc[mask]
    if not ref.empty:
        flag_laps[int(ref['LapNumber'].iloc[0])] = typ

# plot
fig, ax = plt.subplots(figsize=(16, 8))                
ax.set_xlim(0.5, laps['LapNumber'].max() + 0.5)          

# shade flagged laps
for lap, typ in flag_laps.items():
    shade_color = 'red' if typ == 'red' else 'yellow'
    ax.axvspan(lap - 0.5, lap + 0.5, color=shade_color, alpha=0.2, zorder=0)

# plot each driver’s delta line
for drv, df_drv in laps.groupby('Driver'):
    ax.plot(
        df_drv['LapNumber'],
        df_drv['GapToLeader'],
        label=drv,
        color=colors.get(drv, '#666666'),
        linewidth=1,
        zorder=1
    )

# zero‐line
ax.axhline(0, color='black', linestyle='--', linewidth=0.5, zorder=2)

# ticks: one per lap, rotated for readability
ax.set_xticks(range(1, int(laps['LapNumber'].max()) + 1))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# labels & legend
ax.set_title('2025 Canadian GP — Gap to Leader per Lap')
ax.set_xlabel('Lap Number')
ax.set_ylabel('Gap to Leader (s)')
ax.legend(title='Driver', bbox_to_anchor=(1.04, 1), loc='upper left')

plt.tight_layout()
plt.show()
