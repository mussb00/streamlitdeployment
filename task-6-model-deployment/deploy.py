import numpy as np
import pickle
import streamlit as st
import pandas as pd
import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import altair as alt
import collect_new_data
from model_monitoring import residual_chart, actual_vs_predicted_chart, MAPE_chart, residual_distribution_chart
from model_training import model_training, create_lagged_data

# api call + make database
def water_level_predictor(df):
    # click button
    ###############################################
    lag_dates = np.array(df['date'].iloc[-7:])
    lags = np.array(df['belgrade_water_level_cm'].iloc[-7:])
    model = model_training(df)
    water_level = model.predict(np.array(lags).reshape(1, -1))
    rounded_water_level = water_level[0].round()
    return rounded_water_level, lag_dates, lags, model

def main():
    st.title("Flood forecasts in Belgrade, Serbia")
    df = new_data()
    with st.container():
        rounded_water_level = 0
        # button for prediction
        if st.button("water level prediction"):
            rounded_water_level, lag_dates, lags, model = water_level_predictor(df)
            if rounded_water_level < 500:
                st.success(
                    f"water level will be {rounded_water_level}. Nothing to worry about"
                )
            else:
                st.warning(f"water level will be {rounded_water_level}. Be cautious")

            current_date = datetime.datetime.now()
            fig, ax = plt.subplots()
            ax.plot(lag_dates, lags, label="water level")
            ax.scatter(
                (current_date + datetime.timedelta(days=1)).date(),
                rounded_water_level,
                marker="*",
                color="green",
                label="prediction",
            )
            ax.hlines(
                y=500,
                xmin=lag_dates[0],
                xmax=lag_dates[-1],
                color="r",
                linestyle="--",
                label="warning level",
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Water level (cm)")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend()
            st.pyplot(fig)


            n_lag = 7
            target = np.array(df['belgrade_water_level_cm'])
            dates = np.array(df['date'].iloc[n_lag:]) #Formating array of dates in the same way as y vector is.
            
            X, y = create_lagged_data(target, n_lag) #Formating the X matrix and y vector
            test_size = round(1/3 * y.shape[0])
            X_train, y_train, dates_train = X[:-test_size], y[:-test_size], dates[:-test_size]
            X_test, y_test, dates_test = X[-test_size:], y[-test_size:], dates[-test_size:]

            y_hat_train = model.predict(X_train)
            y_hat_test = model.predict(X_test)

            #### Building some plots to monitor the model's quality.
            st.title("Dashboard for Model's Quality Monitoring")
            #Plot of actual versus predicted values
            chart_train = actual_vs_predicted_chart(y_train, y_hat_train, dates_train, is_test_set=False)
            chart_test = actual_vs_predicted_chart(y_test, y_hat_test, dates_test, is_test_set=True)
            # Creates the layout of the plots side by side
            chart_layout = alt.hconcat(chart_train, chart_test)
            # Renders the layout of the plots using Streamlit
            st.write(chart_layout)

            chart_train = residual_chart(y_train, y_hat_train, dates_train, is_test_set=False)
            chart_test = residual_chart(y_test, y_hat_test, dates_test, is_test_set=True)
            chart_layout = alt.hconcat(chart_train, chart_test)
            st.write(chart_layout)

            chart_train = MAPE_chart(y_train, y_hat_train, dates_train, is_test_set=False)
            chart_test = MAPE_chart(y_test, y_hat_test, dates_test, is_test_set=True)
            chart_layout = alt.hconcat(chart_train, chart_test)
            st.write(chart_layout)

            chart_train = residual_distribution_chart(y_train, y_hat_train, is_test_set=False)
            chart_test = residual_distribution_chart(y_test, y_hat_test, is_test_set=True)
            chart_layout = alt.hconcat(chart_train, chart_test)
            st.write(chart_layout)
            
            
@st.cache_data
def new_data():
    with st.spinner('LOADING DATABASE'):
        return collect_new_data.main()
    
    
if __name__ == "__main__":
    main()
