import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt
from sklearn.metrics import mean_absolute_percentage_error

def residual_chart(y, y_hat, array_dates, is_test_set = True):
    residuals = y - y_hat
    data = pd.DataFrame({'residuals':residuals,
                         'date': array_dates
                         })
    subset = 'Test set' if is_test_set else 'Training set'
    chart = alt.Chart(data).mark_circle().encode(
        x=alt.X('date',axis=alt.Axis(title='Date')),
        y=alt.Y('residuals', axis=alt.Axis(title='Error (cm)') ),
        tooltip=['date', 'residuals']
        ).properties(
            title=f'Residual plot (y - y_hat) - {subset}'
            ).interactive()
    return chart

def actual_vs_predicted_chart(y, y_hat, array_dates, is_test_set=True):
    data = pd.DataFrame({'actual': y,
                         'predicted': y_hat,
                         'date': array_dates
                         })
    subset = 'Test set' if is_test_set else 'Training set'
    scatter_plot= alt.Chart(data).mark_point().encode(
        x=alt.X('actual', axis=alt.Axis(title='Actual values (cm)'), scale=alt.Scale(zero=False) ),
        y=alt.Y('predicted', axis=alt.Axis(title='Predicted values (cm)'), scale=alt.Scale(zero=False) ),
        tooltip=['predicted', 'actual']
        ).properties(
            title=f'Actual vs Predicted plot - {subset}'
            ).interactive()

    #Calculating the bissectriz equation:
    m = 1
    b = 0
    data['bissectriz'] = data['actual'].apply(lambda x: m*x + b)

    # Adding one layer for the bissectriz
    bissectriz = alt.Chart(data).mark_line(color='red', opacity=0.5).encode(
        x='actual',
        y='bissectriz'
    )

    # Combining both layers into a single chart
    chart = scatter_plot + bissectriz
    return chart

def MAPE_chart(y, y_hat, array_dates, is_test_set=True):
    data = pd.DataFrame({'actual': y,
                         'predicted': y_hat,
                         'date': array_dates
                         })
    data['mape'] = data.apply(lambda row: 100 * np.abs(row['actual'] - row['predicted']) / row['actual'], axis=1)
    subset = 'Test set' if is_test_set else 'Training set'
    chart= alt.Chart(data).mark_point().encode(
        x=alt.X('date', axis=alt.Axis(title='Date') ),
        y=alt.Y('mape', axis=alt.Axis(title='Mean absolute percentage error (%)') ),
        tooltip=['date', 'mape']
        ).properties(
            title=f'MAPE plot - {subset}'
            ).interactive()
    return chart

def residual_distribution_chart(y, y_hat, is_test_set=True):
    residuals = y - y_hat
    data = pd.DataFrame({'residuals':residuals})
    hist, bin_edges = np.histogram(residuals, bins=6)
    data =pd.DataFrame({'residuals': bin_edges[:-1],
                        'count': hist})
    
    subset = 'Test set' if is_test_set else 'Training set'    
    histogram = alt.Chart(data).mark_bar().encode(
        x=alt.X('residuals', axis=alt.Axis(title='Residuals (cm)'), bin=True), #bin=alt.Bin(bin_size=5)
        y=alt.Y('count', axis=alt.Axis(title='Count') ),
        tooltip=['residuals', 'count']
    ).properties(
        title=f'Residual (y-y_hat) distribution - {subset}'
        ).interactive()
    return histogram
