from mpl_toolkits.mplot3d import Axes3D  
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Improved correlation matrix
import numpy as np  # Linear algebra
import os  # File system access
import pandas as pd  # Data processing, CSV I/O

# ✅ Set the correct file path
file_path = r"C:\Users\ukar1\OneDrive\Pictures\Screenshots\data\iris.csv"

# ✅ Check if the file exists before proceeding
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found. Check the path.")
    exit()

# ✅ Load the dataset
df1 = pd.read_csv(file_path)

# ✅ Assign a dataframe name for reference
df1.dataframeName = 'iris.csv'

# ✅ Print dataset info
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
print(df1.head())

# ✅ Function to plot column distributions
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if 1 < nunique[col] < 50]]  # Select relevant columns
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow  # Integer division

    plt.figure(figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')

    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]

        if not np.issubdtype(columnDf.dtype, np.number):
            columnDf.value_counts().plot.bar()
        else:
            columnDf.hist()

        plt.ylabel('Counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()

# ✅ Enhanced Correlation Matrix with Seaborn
def plotCorrelationMatrix(df):
    filename = getattr(df, "dataframeName", "DataFrame")
    df = df.select_dtypes(include=[np.number])  # Keep only numerical columns

    if df.shape[1] < 2:
        print(f'No correlation plots shown: Only {df.shape[1]} valid columns.')
        return

    plt.figure(figsize=(10, 8))
    corr = df.corr()
    
    # Use Seaborn heatmap for better visualization
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# ✅ Function to plot scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])  # Keep only numerical columns
    df = df.dropna(axis=1)
    df = df[[col for col in df if df[col].nunique() > 1]]

    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values

    for i, j in zip(*np.triu_indices_from(corrs, k=1)):
        ax[i, j].annotate(f'Corr. coef = {corrs[i, j]:.3f}', 
                          (0.8, 0.2), xycoords='axes fraction', 
                          ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')
    plt.show()

# ✅ Generate plots
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1)  # Improved correlation matrix
plotScatterMatrix(df1, 12, 10)
