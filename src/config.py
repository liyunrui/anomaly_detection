class Configs:
    # categorial columns in data
    CATEGORY = ['report_date', 'update_date']

    # aggregate 
    BALANCE_AGG_RECIPE = [
        (["id"], [
                ('new_balance', 'count'),
                ('new_balance', 'min'),
                ('new_balance', 'max'),
                ('new_balance', 'mean'),
                ('new_balance', 'median'),
                ('new_balance', 'var'),
                ('new_balance', 'sum'),
            ]), # how many times the customer has records, and maximum, minimum, median, and mean balance.
        # (["pay_normal"], [
        #         ('new_balance', 'count'),
        #         ('new_balance', 'min'),
        #         ('new_balance', 'max'),
        #         ('new_balance', 'mean'),
        #         ('new_balance', 'median'),
        #         ('new_balance', 'var'),
        #         ('new_balance', 'sum'),
        #     ]), # groupby pay_normal to aggreate his count and maximum, minimum, median, and mean balance.

    ]