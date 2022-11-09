from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import preprocessed

DATA_PATH = Path(__file__).parent / 'audit_data' / 'audit_risk.csv'
COLUMNS = ['LOCATION_ID', 'PARA_A', 'Score_A', 'Risk_A', 'PARA_B', 'Score_B', 'Risk_B', 'TOTAL', 'Money_Value', 'Risk']

def main():
    data = preprocessed(DATA_PATH, COLUMNS)

    risky = data[data.Risk == 1]
    safe = data[data.Risk == 0]
    fig1, axs = plt.subplots(2,2)

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

    fig2, axs3d = plt.subplots(2, projection='3d')
    
    # 3D scatter plot for PARAM_A, Score_A, Risk_A, and Risk
    axs3d[0].scatter(risky.PARAM_A, risky.Score_A, risky.Risk_A color='red', alpha=0.1, label='Risk = 1')
    axs3d[0].scatter(safe.PARAM_A, safe.Score_A, safe.Risk_A color='red', alpha=0.1, label='Risk = 0')

    plt.show()


if __name__ == '__main__':
    main()

