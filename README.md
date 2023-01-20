### Stock Market Prediction Models

**Models** - There will be two models.

1. The first model will provide a classification prediction (binary) - stock market movement prediction model. 
2. The second is a forecast model that will predict the price for the next 12 time segments - stock price forecast model. 

#### Market Prediction

This model uses a number of algortithms (currently only random forest) to make prediction about the direction of return of the very next time step.


Material 

- direction_build_model.ipynb - jupyter notebook that shows the model build process
- dir_test_full_model_serve.ipynb - jupyter notebook that tests logic for model, including preprocessing as well. Lastly, and most important function is to test model endpoint, and show required input to model
- transform_functions - as the name indicates functions used in model build and model predictions
- full_model_serve.py - actual file that houses model and manages prediction requests


#### Market Price Forecast
- forecast_build_model.ipynb - jupyter notebook used to build initial forecasting model
- forecast_test_model.ipoynb - notebook used to test model endpoint as well as basic logic for model serving.
- transform_forecaster.py - functions used transormation for both model build files as well as model predict file
- fct_model_serve.py - actual file that houses model for forecasting and manages prediction requests

### Visualizations

- viz_1.ipynb - notbook used to generate the following visuals (initial rendering)
    1. Forecast based visuals
    2. Past returns vs. Future expected
    3. Return vs volatility
    4. Best and worst buys (bar Chart) of returns ( top 10 / bottom 10)
    5. Technical Analysis - line chart SMA50 / SMA200 by symbol ( add buy /sell signals)
    6. Technical Analysis - buy/sell/hold by stock 