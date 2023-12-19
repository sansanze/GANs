import math
import numpy as np
import statsmodels.api as sm


class Option():
    def __init__(self, type, sort, data, K, r, T, market_price, B=-100):
        self.type = type
        self.sort = sort
        self.data = data
        self.strike = K
        self.barrier = B
        self.risk = r
        self.time = T
        self.market_price = market_price

    def price(self):
        if self.type == 'asian':
            return [self.price_asian()]
        elif self.type == 'european':
            return [self.price_european()]
        elif self.type == 'american':
            # return [self.price_american_avg(), self.price_american_least_squares()]
            return [self.price_american_avg()]
        elif self.type == 'lookback':
            return [self.price_lookback()]
        else:  # Else is barrier
            return [self.price_barrier()]

    def price_european(self):
        last_price = self.data[:, -1]
        if self.sort == 'call':
            simulation_prices = math.exp(-self.risk * self.time) * np.maximum((last_price - self.strike),
                                                                              np.zeros(len(self.data)))

            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]

        elif self.sort == 'put':
            simulation_prices = math.exp(-self.risk * self.time) * np.maximum((self.strike - last_price),
                                                                              np.zeros(len(self.data)))

            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]

        else:
            return print("Error: Invalid parameter")

    def price_american_least_squares(self):
        if self.sort == 'call':
            # At maturity we have the discounted cash flow, current cash flow, and optimal exercise date at T

            discounted_future_cashflow = np.maximum((self.data[:, -1] - self.strike), np.zeros(
                len(self.data)))  # We do not discount here yet, but downbelow
            optimal_cash_flow = np.maximum((self.data[:, -1] - self.strike), np.zeros(len(self.data)))
            optimal_date = np.repeat(self.time, len(optimal_cash_flow))

            for i in range(2, len(self.data[0, :])):  # Fixed loop range
                discounted_future_cashflow = np.exp(-self.risk) * discounted_future_cashflow
                current_cashflow = np.maximum((self.data[:, -i] - self.strike), np.zeros(len(self.data)))
                current_asset_price = self.data[:, -i]  # Fixed indexing

                # Train model on data only with positive cash flows at current state: value_at_i
                pos_future_cashflow = []
                pos_current_prices = []
                pos_current_cashflow = []

                for j in range(len(current_cashflow)):
                    if current_cashflow[j] != 0:
                        pos_future_cashflow.append(discounted_future_cashflow[j])
                        pos_current_prices.append(current_asset_price[j])
                        pos_current_cashflow.append(current_cashflow[j])

                if len(pos_current_cashflow) > 1:
                    pos_current_prices_squared = np.square(pos_future_cashflow)
                    pos_current_prices = sm.add_constant(pos_current_prices)

                    # Train model on real future cash flow and current index price
                    model = sm.OLS(pos_future_cashflow, np.column_stack(
                        (pos_current_prices, pos_current_prices_squared)))  # Corrected regression inputs
                    result = model.fit()

                    # Predict future expected cash flow for all price paths using the current index price
                    predictions = []
                    for k in range(len(discounted_future_cashflow)):
                        # Discount future cash flow
                        predictions.append(np.exp(-self.risk) *
                                           result.predict(np.array([[1, self.data[k, -i], self.data[k, -i] ** 2]]))[0])

                    for k in range(len(predictions)):
                        if predictions[k] < discounted_future_cashflow[k]:
                            optimal_cash_flow[k] = discounted_future_cashflow[k]
                            optimal_date[k] = self.time - i + 1

            # For every price path the cash flow and date has been calculated
            simulation_prices = np.exp(-self.risk * optimal_date) * optimal_cash_flow

            # Results
            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]

        elif self.sort == 'put':
            # At maturity we have the discounted cash flow, current cash flow, and optimal exercise date at T

            discounted_future_cashflow = np.maximum((self.strike - self.data[:, -1]), np.zeros(
                len(self.data)))  # We do not discount here yet, but downbelow
            optimal_cash_flow = np.maximum((self.strike - self.data[:, -1]), np.zeros(len(self.data)))
            optimal_date = np.repeat(self.time, len(optimal_cash_flow))

            for i in range(2, len(self.data[0, :])):  # Fixed loop range
                discounted_future_cashflow = np.exp(-self.risk) * discounted_future_cashflow
                current_cashflow = np.maximum((self.strike - self.data[:, -i]), np.zeros(len(self.data)))
                current_asset_price = self.data[:, -i]  # Fixed indexing

                # Train model on data only with positive cash flows at current state: value_at_i
                pos_future_cashflow = []
                pos_current_prices = []
                pos_current_cashflow = []

                for j in range(len(current_cashflow)):
                    if current_cashflow[j] != 0:
                        pos_future_cashflow.append(discounted_future_cashflow[j])
                        pos_current_prices.append(current_asset_price[j])
                        pos_current_cashflow.append(current_cashflow[j])

                if len(pos_current_cashflow) > 1:
                    pos_current_prices_squared = np.square(pos_future_cashflow)
                    pos_current_prices = sm.add_constant(pos_current_prices)

                    # Train model on real future cash flow and current index price
                    model = sm.OLS(pos_future_cashflow, np.column_stack(
                        (pos_current_prices, pos_current_prices_squared)))  # Corrected regression inputs
                    result = model.fit()

                    # Predict future expected cash flow for all price paths using the current index price
                    predictions = []
                    for k in range(len(discounted_future_cashflow)):
                        # Discount future cash flow
                        predictions.append(np.exp(-self.risk) *
                                           result.predict(np.array([[1, self.data[k, -i], self.data[k, -i] ** 2]]))[0])

                    for k in range(len(predictions)):
                        if predictions[k] < discounted_future_cashflow[k]:
                            optimal_cash_flow[k] = discounted_future_cashflow[k]
                            optimal_date[k] = self.time - i + 1

            # For every price path the cash flow and date has been calculated
            simulation_prices = np.exp(-self.risk * optimal_date) * optimal_cash_flow

            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]

        else:
            return print("Error: Invalid parameter")

    def price_american_avg(self):
        if self.sort == 'call':
            lower = math.exp(-self.risk * self.time) * (
                np.maximum((self.data[:, -1] - self.strike), np.zeros(len(self.data))))
            upper = np.maximum((self.data[:, -1] - self.strike), np.zeros(len(self.data)))
            simulation_prices = (lower + upper) / 2

            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]

        elif self.sort == 'put':
            lower = math.exp(-self.risk * self.time) * (
                np.maximum((self.strike - self.data[:, -1]), np.zeros(len(self.data))))
            upper = np.maximum((self.strike - self.data[:, -1]), np.zeros(len(self.data)))
            simulation_prices = (lower + upper) / 2

            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]

        else:
            return print("Error: Invalid parameter")

    def price_asian(self):
        avg_price = np.mean(self.data, axis=1)
        if self.sort == 'call':
            simulation_prices = math.exp(-self.risk * self.time) * np.maximum((avg_price - self.strike),
                                                                              np.zeros(len(self.data)))
            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]
        elif self.sort == 'put':
            simulation_prices = math.exp(-self.risk * self.time) * np.maximum((self.strike - avg_price),
                                                                              np.zeros(len(self.data)))
            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]
        else:
            return print("Error: Invalid parameter")

    def price_lookback(self):
        max_price = np.max(self.data, axis=1)
        min_price = np.min(self.data, axis=1)
        if self.sort == 'call':
            simulation_prices = math.exp(-self.risk * self.time) * np.maximum((max_price - self.strike),
                                                                              np.zeros(len(self.data)))
            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]
        elif self.sort == 'put':
            simulation_prices = math.exp(-self.risk * self.time) * np.maximum((self.strike - max_price),
                                                                              np.zeros(len(self.data)))
            # Results
            option_price = np.mean(simulation_prices)  # Mean estimated price
            error = abs((option_price - self.market_price) / self.market_price)
            return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]
        else:
            return print("Error: Invalid parameter")

    def price_barrier(self):
        payoffs = np.zeros(len(self.data))

        for i, row in enumerate(self.data):
            max_price = np.max(row)
            min_price = np.min(row)

            if (self.type == 'upout' and max_price < self.barrier) or \
                    (self.type == 'downout' and min_price > self.barrier) or \
                    (self.type == 'upin' and max_price > self.barrier) or \
                    (self.type == 'downin' and min_price < self.barrier):
                if self.sort == 'call':
                    payoffs[i] = max(row[-1] - self.strike, 0)
                elif self.sort == 'put':
                    payoffs[i] = max(self.strike - row[-1], 0)

        # Results
        option_price = math.exp(-self.risk * self.time) * np.mean(payoffs)  # Mean estimated price
        error = abs((option_price - self.market_price) / self.market_price)
        return [self.type, self.sort, error, self.market_price, self.strike, self.time, option_price]
