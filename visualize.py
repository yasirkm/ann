from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import preprocessed

DATA_PATH = Path(__file__).parent / 'audit_data' / 'audit_risk.csv'
COLUMNS = ['LOCATION_ID', 'PARA_A', 'Score_A', 'Risk_A', 'PARA_B', 'Score_B', 'Risk_B', 'TOTAL', 'Money_Value', 'Risk']

def main():
    show_graphs()
    show_graphs(preprocess=True)

def show_graphs(data_path=DATA_PATH, cols=COLUMNS, preprocess=False):
    '''
        Shows 2D and 3D graph of audit_risk data
    '''
    # Reading data file
    if preprocess:
        data = preprocessed(data_path, cols)
    else:
        data = pd.read_csv(data_path)
    
    # Separating data to risky and safe date with the use of the Risk column
    risky = data[data.Risk == 1]
    safe = data[data.Risk == 0]
    fig, axs = plt.subplots(2,2)
    fig.suptitle("2D Graphs")

    # Scatterplot for TOTAL, Money_Value, and Risk
    axs[0][0].scatter(risky.TOTAL, risky.Money_Value, color='red', alpha=0.1, label='Risk = 1')
    axs[0][0].scatter(safe.TOTAL, safe.Money_Value, color='blue', alpha=0.1, label='Risk = 0')
    axs[0][0].set_title('Scatter plot for TOTAL, Money_value, and Risk')
    axs[0][0].set_ylabel('Money_Value')
    axs[0][0].set_xlabel('TOTAL')
    axs[0][0].legend()

    # Bar plot for LOCATION_ID and Risk
    categories = sorted(data.LOCATION_ID.unique())
    width = 0.5
    x_placements =np.arange(len(categories))
    axs[0][1].bar(x_placements-width/2, list(map(len, (risky[risky.LOCATION_ID==category] for category in categories))), width, color='red', label='Risk = 1')
    axs[0][1].bar(x_placements+width/2, list(map(len, (safe[safe.LOCATION_ID==category] for category in categories))), width, color='blue', label='Risk = 0')
    axs[0][1].set_title('Bar plot for LOCATION_ID and Risk')
    axs[0][1].set_xticks(x_placements)
    axs[0][1].set_xticklabels(categories)
    axs[0][1].set_xlabel('LOCATION_ID')
    axs[0][1].legend()

    # Box plot for TOTAL and Risk
    bplot = axs[1][0].boxplot((risky.TOTAL,safe.TOTAL), vert=False, patch_artist=True, labels=('Risk = 1', 'Risk = 0'))
    for patch, color in zip(bplot['boxes'], ('red', 'blue')):
        patch.set_facecolor(color)

    # Box plot for Money_Value and Risk
    bplot = axs[1][1].boxplot((risky.Money_Value,safe.Money_Value), vert=False, patch_artist=True, labels=('Risk = 1', 'Risk = 0'))
    for patch, color in zip(bplot['boxes'], ('red', 'blue')):
        patch.set_facecolor(color)

    fig3d, axs3d = plt.subplots(2, subplot_kw={'projection':'3d'})
    fig3d.suptitle('3D Graphs')
    
    # 3D scatter plot for PARA_A, Score_A, Risk_A, and Risk
    axs3d[0].scatter(risky.PARA_A, risky.Score_A, risky.Risk_A, color='red', alpha=0.3, label='Risk = 1')
    axs3d[0].scatter(safe.PARA_B, safe.Score_B, safe.Risk_B, color='blue', alpha=0.3, label='Risk = 0')

    # 3D scatter plot for PARA_B, Score_B, Risk_B, and Risk
    axs3d[1].scatter(risky.PARA_B, risky.Score_B, risky.Risk_B, color='red', alpha=0.3, label='Risk = 1')
    axs3d[1].scatter(safe.PARA_B, safe.Score_B, safe.Risk_B, color='blue', alpha=0.3, label='Risk = 0')
    
    for ax3d, char in zip(axs3d, ('A','B')):
        ax3d.set_xlabel(f'PARA_{char}')
        ax3d.set_ylabel(f'Score_{char}')
        ax3d.set_zlabel(f'Risk_{char}')
        ax3d.legend()

    plt.show()


if __name__ == '__main__':
    main()