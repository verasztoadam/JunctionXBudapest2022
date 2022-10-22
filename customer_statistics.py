import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    df_customers = pd.read_csv('./data/gen/customers.csv')
    print(df_customers.describe().to_string())

    """
    Generate scatter plot that show the number of checkouts according to the coupon counts,
    the color indicates the number of occurrences
    """
    groups = df_customers["checkout_count"].unique()
    plot_data = df_customers.groupby("checkout_count")["coupon_count"].value_counts().items()
    x = []
    y = []
    c = []
    for d in plot_data:
        x.append(d[0][0])
        y.append(d[0][1])
        c.append(d[1] ** (1 / 10))

    plt.plot([0, 80], [0, 80], color='r')
    plt.plot([0, 100], [0, 41], color='b')
    plt.scatter(x, y, c=c, cmap='turbo', zorder=2)
    color_bar = plt.colorbar()
    color_bar.set_ticks([])
    color_bar.ax.get_yaxis().labelpad = 10
    color_bar.ax.set_ylabel('=> Number of Occurences =>', rotation=90)
    plt.xlabel("Number of checkouts")
    plt.ylabel("Number of coupon uses")
    plt.title("Coupon usage in the function of checkouts")

    """Checkout frequency diagram"""
    plt.figure()
    checkout_freq = []
    for i in range(max(df_customers.checkout_count.tolist())):
        count = 0
        for element in df_customers.checkout_count.tolist():
            if element == i:
                count += 1
        checkout_freq.append(count)

    coupon_freq = []
    for i in range(max(df_customers.coupon_count.tolist())):
        count = 0
        for element in df_customers.coupon_count.tolist():
            if element == i:
                count += 1
        coupon_freq.append(count)

    common_length = min(len(checkout_freq), len(coupon_freq))
    checkout_freq = checkout_freq[:common_length]
    coupon_freq = coupon_freq[:common_length]
    x = np.arange(common_length)

    width = 0.5
    rects_1 = plt.bar(x - width / 2, checkout_freq, width, label='Checkout frequency')
    rects_2 = plt.bar(x + width / 2, coupon_freq, width, label='Coupons frequency')
    plt.yscale("log")
    plt.xlabel("Number of checkouts")
    plt.ylabel("Number of occurrences")
    plt.legend()
    plt.title("Frequency of normal and coupon checkout numbers")
    plt.tight_layout()

    """
    Plot frequency of ratio of coupon usage.
    """
    X_RESOLUTION = 50
    checkout_percent = [0] * X_RESOLUTION

    checkout_count_list = df_customers.checkout_count.tolist()
    coupons_count_list = df_customers.coupon_count.tolist()
    customers_count = len(df_customers["customer"].tolist())

    for i in range(len(checkout_count_list)):
        for j in range(X_RESOLUTION):
            if checkout_count_list[i] != 0:
                if (coupons_count_list[i] / checkout_count_list[i]) <= ((j + 1) / X_RESOLUTION):
                    checkout_percent[j] += (1 / customers_count)
                    break
            else:
                break

    plt.figure()
    plt.xlabel("Coupon usage percentage")
    plt.ylabel("Occurrence percentage")
    plt.title("Coupon usage median percentages over customers")
    plt.plot(np.arange(0, 100, 100 / X_RESOLUTION), checkout_percent)

    """
    Plot frequency of ratio of coupon usage for higher checkout count.
    """
    X_RESOLUTION = 20
    CHECK_OUT_THRESHOLD = 10
    checkout_percent = [0] * X_RESOLUTION

    checkout_count_list = df_customers[df_customers["checkout_count"] >= CHECK_OUT_THRESHOLD].checkout_count.tolist()
    coupons_count_list = df_customers[df_customers["checkout_count"] >= CHECK_OUT_THRESHOLD].coupon_count.tolist()
    customers_count = len(df_customers[df_customers["checkout_count"] >= CHECK_OUT_THRESHOLD]["customer"].tolist())

    for i in range(len(checkout_count_list)):
        for j in range(X_RESOLUTION):
            if checkout_count_list[i] != 0:
                if (coupons_count_list[i] / checkout_count_list[i]) <= ((j + 1) / X_RESOLUTION):
                    checkout_percent[j] += (1 / customers_count)
                    break
            else:
                break

    plt.figure()
    plt.xlabel("Coupon usage percentage")
    plt.ylabel("Occurrence percentage")
    plt.title("Coupon usage median percentages over customers with over 10 checkouts")
    plt.plot(np.arange(0, 100, 100 / X_RESOLUTION), checkout_percent)

    """Plot correlation matrix"""
    FEATURE_COUNT = 10

    corr = df_customers.corr()
    plt.figure(num=None, figsize=(FEATURE_COUNT, FEATURE_COUNT), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=5)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix', fontsize=15)

    plt.show()
