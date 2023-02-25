import numpy as np
import pickle
import streamlit as st
import pandas as pd
import requests
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

@st.cache_resource
def load_model():
    model=pickle.load(open("/app/streamlitdeployment/task-6-model-deployment/linear-regression.pkl", "rb"))
    return model


# api call + make database
def water_level_predictor():
    # click button
    response = requests.get(
        "https://www.hidmet.gov.rs/eng/hidrologija/godisnje/godisnjak.php?sifra=45099"
    )
    data = response.content.decode("utf-8")
    soup = BeautifulSoup(data, "html.parser")

    table_rows = soup.find("div", {"id": "sadrzaj"}).find("table").find_all("tr")
    monthly_data = pd.DataFrame(columns=[f"month_{i}" for i in range(1, 13)])

    # populate dataframe with monthly water level data
    for idx, row in enumerate(table_rows):
        if idx > 2:
            columns = row.find_all("td")

            data = {
                "month_1": columns[1],
                "month_2": columns[2],
                "month_3": columns[3],
                "month_4": columns[4],
                "month_5": columns[5],
                "month_6": columns[6],
                "month_7": columns[7],
                "month_8": columns[8],
                "month_9": columns[9],
                "month_10": columns[10],
                "month_11": columns[11],
                "month_12": columns[12],
            }
            monthly_data = monthly_data.append(data, ignore_index=True)

    # generate lag dates
    lag_dates = [
        datetime.datetime.now() - datetime.timedelta(days=i) for i in range(1, 8)
    ]

    # get past week water levels
    lags = []
    for date in lag_dates:
        lag = int(monthly_data[f"month_{date.month}"].iloc[date.day].text.strip())
        lags.append(lag)
    
    model=load_model()
    water_level = model.predict(np.array(lags).reshape(1, -1))
    rounded_water_level = water_level[0].round()

    return rounded_water_level, monthly_data, lag_dates, lags


def main():
    st.title("Flood forecasts in Belgrade, Serbia")

    with st.container():
        rounded_water_level = 0
        # button for prediction
        if st.button("water level prediction"):
            rounded_water_level, monthly_data, lag_dates, lags = water_level_predictor()

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
                current_date,
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
            ax.set_ylabel("Water level (m)")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend()
            st.pyplot(fig)


if __name__ == "__main__":
    main()
