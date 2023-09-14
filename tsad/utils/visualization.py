from typing import List, Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import missingno as msno
from matplotlib import pyplot as plt



def plot_signals(df_signals: pd.DataFrame,
                 use_resampler: bool = True,
                 scale: Optional[str] = None,
                 scale_columns: Optional[List[str]] = None,
                 show: bool = True) -> go.Figure:
    """
    Plot time series signals from a DataFrame.

    Args:
        df_signals (pd.DataFrame): The DataFrame containing time series signals.
        use_resampler (bool, optional): Whether to use plotly_resampler for interactive zooming (default is True).
        scale (str, optional): Scaling method for the signals. Options are 'minmax' and 'standard' (default is None).
        scale_columns (list of str, optional): List of specific columns to scale (default is None).
        show (bool, optional): Whether to display the plot (default is True).

    Returns:
        go.Figure: A Plotly figure displaying the time series signals.

    Examples:
        # Plot all numeric columns with default settings
        fig = plot_signals(df)

        # Plot specific columns with min-max scaling
        fig = plot_signals(df, scale='minmax', scale_columns=['signal1', 'signal2'])

        # Plot without interactive zooming
        fig = plot_signals(df, use_resampler=False)

    Additional Information:
        - `scale` (str): Choose a scaling method for the signals:
          - 'minmax': Scales all numeric columns between 0 and 1.
          - 'standard': Standardizes all numeric columns (mean=0, std=1).
          Default is `None`.

        - `scale_columns` (list of str): Specify a subset of columns to apply scaling, leaving others unchanged.
          Default is `None`.

        - `show` (bool): Set this to `True` if you want to display the plot. Default is `True`.

        - The generated Plotly figure can be further customized and saved as needed.
    """
    df = df_signals.copy(deep=True)
    df = df.select_dtypes(include=['number',])
    numeric_columns = df.columns
    if scale_columns is not None:
        scale_columns = [col for col in scale_columns if col in numeric_columns]
    else:
        scale_columns = numeric_columns
    
    if scale == 'minmax':
        scaler = MinMaxScaler()
        df[scale_columns] = scaler.fit_transform(df)
    elif scale == 'standard':
        scaler = StandardScaler()
        df[scale_columns] = scaler.fit_transform(df)

    fig = px.line(df, x=df.index, y=numeric_columns)
    fig.update_layout(legend=dict(
        orientation="h",
        y=-0.5,
    ),
        height=500)
    if use_resampler:
        fig = FigureResampler(fig)
        if show:
            fig.show_dash(mode='inline')
        return fig
    else:
        if show:
            fig.show()
        return fig
    

def plot_missing_values(dataframe):
    """
    Create a matrix plot of missing values in a DataFrame.

    This function uses the `missingno` library to create a matrix plot
    that visualizes missing values in the DataFrame. If the date range
    is small (within the same day), it adds time to the labels.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes containing the plot.

    Example:
        import pandas as pd
        import matplotlib.pyplot as plt

        # Create a sample DataFrame with missing values
        data = {'A': [1, 2, None, 4, 5],
                'B': [None, 2, 3, 4, 5],
                'C': [1, 2, 3, 4, 5]}
        df = pd.DataFrame(data)

        # Plot missing values
        ax = plot_missing_values(df)
        plt.show()
    """
    ax = msno.matrix(dataframe)
    
    # Check if the date range is small (within the same day)
    date_range = dataframe.index[-1] - dataframe.index[0]
    small_date_range = date_range.total_seconds() < 86400  # 86400 seconds in a day
    
    # Set yticks to automatically scale to the time range of the DataFrame
    num_ticks = min(dataframe.shape[0], 10)  # Set the maximum number of ticks to 10
    ticks = np.linspace(0, dataframe.shape[0] - 1, num=num_ticks, dtype=int)
    
    labels = [s.strftime("%Y-%m-%d %H:%M:%S") if small_date_range else s.strftime("%Y-%m-%d") for s in dataframe.index[ticks]]
    plt.yticks(ticks=ticks, labels=labels)
    plt.title('Missing values')

    return ax