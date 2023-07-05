import os, logging

import metrics

import pandas as pd
from neuralprophet import NeuralProphet
from neuralprophet import set_log_level as set_np_log_level

set_np_log_level("ERROR")

def read_inputs(path):
    
    train = pd.read_csv(os.path.join(path, "train.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))
    
    return train, test

def format_df(df):

    # datetime dtype can't be specified during reading so we have to convert here
    df["day_id"] = pd.to_datetime(df["day_id"])
    # concat ts identifiers into unique_id
    df["ID"] = df["but_num_business_unit"].astype(str) + "_" + df["dpt_num_department"].astype(str)
    # drop unnecessary columns
    df.drop(columns=["but_num_business_unit", "dpt_num_department"], inplace=True)
    # apply prophet format
    df = df.rename(columns={"day_id":"ds", "turnover":"y"})

    return df

def get_max_length_ids(df):

    # count nb of dates by id
    g = df.groupby("ID")["ds"].count().reset_index()
    # get ids with only max nb of dates available
    ids_to_keep = g[g["ds"] == g["ds"].max()]["ID"].to_list()

    return ids_to_keep
    

def split_df(df, validation_size):

    # split using cutoff date since we have a dataframe in the "long" format
    cutoff = df["ds"].unique()[-validation_size]
    train = df[df["ds"] < cutoff]
    val = df[df["ds"] >= cutoff]

    return train, val

def process_inputs(train, test):

    # normalize format
    train = format_df(train)
    test = format_df(test)

    # groupby unique id
    train = train.groupby(["ds", "ID"])["y"].sum().reset_index()
    
    # make sure all time series have same length
    id_to_keep = get_max_length_ids(train)
    train = train[train["ID"].isin(id_to_keep)]
    test = test[test["ID"].isin(id_to_keep)]

    # cut a validation set out of train to evaluate our model
    train, val = split_df(train, 8)

    return train, val, test

def fit_and_predict(params, history, future):

    m = NeuralProphet(**params)
    m.fit(history, early_stopping=True, progress="off")
    forecast = m.predict(pd.concat([history, future]))
    forecast = forecast[forecast["ds"] > history["ds"].max()]

    return forecast

def evaluate_forecast(validation_df, eval_df):

    global_rmse = metrics.rmse(validation_df["y"].values, eval_df["yhat1"].values)

    return global_rmse

def write_output(df, path):
    df.to_csv(path, index=False)

def pipeline(inputs_path, output_path, model_params):

    # Input
    train, test = read_inputs(inputs_path)
    train, validation, test = process_inputs(train, test)

    # Model evaluation and prediction
    eval_forecast = fit_and_predict(model_params, history=train, future=validation)
    scores = evaluate_forecast(validation, eval_forecast)
    logging.info(scores)
    forward_forecast = fit_and_predict(model_params, history=pd.concat([train, val]), future=test)

    # Output
    write_output(forward_forecast, output_path)

if __name__ == "__main__":

    # pipeline setup
    inputs_path = "./data"
    output_path = "forecast.csv"
    model_params = {
        "daily_seasonality":False,
        "weekly_seasonality":False,
        "yearly_seasonality":True,
        "season_global_local":"local",
        "n_changepoints":8,
        "trend_global_local":"local",
        "n_lags":4,
        "unknown_data_normalization":True
    }

    # run pipeline
    pipeline(inputs_path, output_path, model_params)