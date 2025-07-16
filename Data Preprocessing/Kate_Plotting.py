import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
import os
import operator



def MedianWardPrices():
    # plots all wards median house prices over 1995-2023
    # it's really messy and we may be better off picking a handful of
    # representitive wards
    
    # ensures the correct path is chosen whatever system i work on
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'Data', 'Wards', 'Median_Prices.csv'), index_col=0)
    dataInf = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                       'Data', 'UK', 'Inflation.csv'), index_col=0)


    # the time is probably wrong but itll only be wrong by ~6 months so not 
    # that big a deal
    time = np.linspace(1995, 2023, 110)
    # timeInf = list(map(lambda x: int(x), dataInf.iloc[:, 0]))

    pricesMed = data
    # inflationRate = (dataInf.iloc[:, 1] + 100) / 100



    # plotting median of medians, min and max, and std shading
    # this could (and should) be implemented with pandas
    med, small, big, std = np.zeros(110), np.zeros(110), np.zeros(110), np.zeros(110)
    for date in range(0, 110):
        pricesTemp = pricesMed.iloc[:, date]
        med[date] = (np.mean(pricesTemp))
        small[date] = (min(pricesTemp))
        big[date] = (max(pricesTemp))
        std[date] = (np.std(pricesTemp))

    # calculate lines for the standard deviation shading
    stdupper = med + std
    stdlower = med - std

    # inflation = np.zeros(29)
    # inflation[0] = med[0]
    # for x in range(0, 28):
    #     inflation[x] = (inflation[-1]*inflationRate.iloc[x])

    fig, ax = plt.subplots(figsize=(10, 6))

    # linear approximation of prices post 2009
    linearApprox = Polynomial.fit(time[55:], med[55:], 1)
    # ax.plot([2009, 2023], [linearApprox(2009), linearApprox(2023)], label='Linear Approx. of Price')
    # print((linearApprox(2023)-linearApprox(2009))/14)


    # price plotting
    ax.plot(time, med, label='Mean Price', color='red')
    ax.plot(time, small, label='Minimum Price', color='blue')
    ax.plot(time, big, label='Maximum Price', color='teal')
    ax.fill_between(time, stdupper, stdlower, alpha=0.1,
                    label='Standard Deviation', color='red')

    initDifference = big[0] - small[0]
    finalDifference = big[-1] - small[-1]
    print(f'1995 difference: {initDifference}\n2023 difference: {finalDifference}')

    # inflation plotting
    # ax.plot(timeInf, inflation)

    # 2008 financial crisis marker
    ax.axvline(x=2008, linestyle='--', color='orange', label='2008-2009 Financial Crisis')
    ax.axvline(x=2009, linestyle='--', color='orange')

    ax.set_xlim([1995, 2023])
    ax.set_xlabel('Year')
    ax.set_ylabel('Price (£)')
    ax.set_title('Median price of housing in Bristol by ward')
    ax.legend(loc='upper left')
    fig.show()
    ax.grid()
    print(small)

    fig.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'ward mean prices.pdf'), format='pdf')


def EnglandWalesMedianPrices():
    # plots the median price for all england and wales regions along with the
    # mean of the bristol medians
    bristolData = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data',
                              'Wards', 'Median_Prices.csv'), index_col=0)
    
    ukData = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data',
                         'UK', 'Median_Prices.csv'), index_col=0)
    inflation = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data',
                            'UK', 'Inflation.csv'), index_col=0)
    
    bristolData = bristolData.mean()
    englandMean = ukData.iloc[:-2, :].mean()
    englandSTD = ukData.iloc[:-2, :].std()
    walesData = ukData.iloc[-2, :]
    londonData = ukData.iloc[-1, :]
    
    inflation = inflation.map(lambda x: 1+(float(x)/100)).to_numpy()


    time = np.linspace(1995, 2023, 110)
    
    inflations = [(bristolData.iloc[0]+englandMean.iloc[0]+walesData.iloc[0]+londonData.iloc[0])/4]
    for x in range(0, len(inflation)):
        inflations.append(inflations[-1]*inflation[x][0])

    fig, ax = plt.subplots(figsize=(10, 6))

    # plotting location mean
    ax.plot(time, bristolData, label='Bristol', color='green')
    ax.plot(time, englandMean, label='England', color='red')
    ax.fill_between(time, englandMean+englandSTD, englandMean-englandSTD, alpha=0.1, color='red', label='England Standard Deviation')
    ax.plot(time, walesData, label='Wales', color='blue')
    ax.plot(time, londonData, label='London', color='teal')

    # plotting financial crisis and inflation
    ax.plot(range(1995, 2025), inflations, label='Inflation (CPI)', color='black')
    ax.axvline(2008, ls='--', color='orange')
    ax.axvline(2009, ls='--', color='orange', label='2008-2009 Financial Crisis')

    # other aesthetic components
    ax.set_title('Mean price of housing in the UK')
    ax.set_xlim([1995, 2023])
    ax.set_xlabel('Year')
    ax.set_ylabel('Price (£)')
    ax.grid()
    ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(os.path.dirname(__file__), 'Figures', 'mean price of housing uk.pdf'), format='pdf')


def HouseTypeSales():
    newBuild = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'Data', 'Bristol_New_Sales.csv'))
    oldBuild = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'Data', 'Bristol_Old_Sales.csv'))

    ukNew = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                     'Data', 'Raw', 'UK_Sales_Of_Newly_Built.csv'),
                        thousands=',')
    ukOld = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                     'Data', 'Raw', 'UK_Sales_Of_Old_Built.csv'),
                        thousands=',')

    newBuild = newBuild.iloc[:, 4:]
    oldBuild = oldBuild.iloc[:, 4:]

    # group uk data by local authority and calculate uk (minus london) average
    # londonNew = ukNew[ukNew['Local authority name'] == 'City of London'].iloc[:, 4:]
    # londonOld = ukOld[ukOld['Local authority name'] == 'City of London'].iloc[:, 4:]

    # ukNew = ukNew[ukNew['Local authority name'] != 'City of London']
    # ukOld = ukOld[ukOld['Local authority name'] != 'City of London']
    ukNewGroup = ukNew.drop(['Local authority code', 'Ward code', 'Ward name'], axis=1).groupby('Local authority name').sum()
    ukOldGroup = ukOld.drop(['Local authority code', 'Ward code', 'Ward name'], axis=1).groupby('Local authority name').sum()
    
    ukNewMean = ukNewGroup.mean()
    ukNewSTD = ukNewGroup.std()
    ukOldMean = ukOldGroup.mean()
    ukOldSTD = ukOldGroup.std()



    # calculate mean and std for the new and old data
    time = np.linspace(1996, 2023, 110)
    newTotal, oldTotal, newSTD, oldSTD = np.zeros(110), np.zeros(110), np.zeros(110), np.zeros(110)
    for date in range(0, 110):
        newTemp = newBuild.iloc[:, date]
        oldTemp = oldBuild.iloc[:, date]
        newTotal[date] = sum(newTemp)
        newSTD[date] = np.std(newTemp)
        oldTotal[date] = sum(oldTemp)
        oldSTD[date] = np.std(oldTemp)


    fig, ax = plt.subplots(ncols=2)

    ax[0].plot(time, newTotal, label='Bristol', color='teal')
    ax[0].plot(time, ukNewMean, label='Uk Mean', color='red')
    ax[0].set_title('Number of New Build Properites Sold')
    ax[0].fill_between(time, ukNewMean+ukNewSTD, ukNewMean-ukNewSTD, alpha=0.1, color='red')
    ax[0].axvline(2008, ls='--', label='2008-2009 Financial Crisis')
    ax[0].axvline(2009, ls='--')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(time, oldTotal, label='Bristol', color='teal')
    ax[1].plot(time, ukOldMean, label='Uk Mean', color='red')
    ax[1].fill_between(time, ukOldMean+ukOldSTD, ukOldMean-ukOldSTD, alpha=0.1, color='red')
    ax[1].axvline(2008, ls='--', label='2008-2009 Financial Crisis')
    ax[1].axvline(2009, ls='--')
    ax[1].set_title('Number of Old Build Properties Sold')
    ax[1].legend()
    ax[1].grid()


def NetDwellings():

    # data comes from gov.uk ministry of housing
    englandData = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data',
                                      'Raw', 'England_Net_housing.csv'),
                         thousands=',')

    bristolData = englandData[englandData['Authority Data'] == 'Bristol, City of UA'].iloc[0, 2:]
    londonData = englandData[englandData['Authority Data'] == 'London Boroughs'].iloc[0, 2:]/13
    englandMean = englandData[englandData['Authority Data'] != 'London Boroughs'].iloc[:, 2:].mean()
    englandSTD = englandData[englandData['Authority Data'] != 'London Boroughs'].iloc[:, 2:].std()

    time = np.linspace(2002, 2024, num=23)

    fig, ax = plt.subplots()
    ax.plot(time, bristolData, label='Bristol', color='teal')
    ax.plot(time, londonData, label='London Borough Average', color='green')
    ax.plot(time, englandMean, label='England Average', color='red')
    ax.fill_between(time, englandMean+englandSTD, englandMean-englandSTD, alpha=0.1, color='red')
    ax.axvline(2008, ls='--', label='2008-2009 Financial Crisis')
    ax.axvline(2009, ls='--')

    ax.set_title('Net Additional Dwellings')
    ax.legend()
    ax.grid()


def DistanceCorrelation():
    # performs a correlation test on ward prices, then another on that data with
    # distances

    bristolData = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data',
                              'Wards', 'Median_Prices.csv'),
                              index_col=0)

    # ward distances
    distanceData = pd.read_csv(os.path.join(os.path.dirname(__file__),
                               'distance_matrix.csv'), index_col=0)

    corr95 = bristolData.transpose().corr() # transpose for pandas to correlate over rows
    print(corr95['Clifton'])
    wardCount = corr95.shape[0]

    # normalising distance data & arranging inline with price data
    distanceData = distanceData.reindex(list(bristolData.index.values))
    distanceData = distanceData[list(bristolData.index.values)]

    distanceData = distanceData.map(lambda x: x/distanceData.max(axis=None))
    distanceData = distanceData.map(lambda x: abs(distanceData.max(axis=None)-x))

    # enacting the inverse square law on the data (optional)
    distanceData = distanceData.map(lambda x: pow(x, 2))

    # plot 2 heatmaps of correlation and distance
    fig, ax = plt.subplots(ncols=2)
    im0 = ax[0].imshow(corr95)
    ax[0].set_xticks(range(wardCount), labels=list(corr95), rotation=90)
    ax[0].set_yticks(range(wardCount), labels=list(corr95))
    ax[0].set_title('House Price Correlation By Ward')
 
    im1 = ax[1].imshow(distanceData)
    ax[1].set_xticks(range(distanceData.shape[0]), labels=list(distanceData), rotation=90)
    ax[1].set_yticks(range(distanceData.shape[0]), labels=list(distanceData))
    # ax[1].set_yticks([])
    ax[1].set_title('Normalised Distance Between Wards (Higher is Closer)')

    scale1 = fig.colorbar(im1)  # need to make the color bar not change the size of the second plot

    # calculate and plot distance, ward price correlaion as a bar chart
    fig, ax = plt.subplots()
    distancePriceCorr = np.zeros(wardCount)
    for ward in range(wardCount):
        distancePriceCorr[ward] = distanceData.iloc[ward, :].corr(corr95.iloc[ward, :])


    ax.bar(list(distanceData), distancePriceCorr, color='teal')
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Ward Price Distance Correlation Coefficients')
    ax.set_ylabel('Pearson Correlation Coefficient')


def DistanceCorrelation2():
    def TimeSeriesPlotting(bristolData, distanceData):
        # calculating correlation coefficients for every year to determine if
        # the correlation is consistent
        # ward: which ward to plot data for, 7 is clifton, 14 is central, 24 is lawrence hill
        # show to TA, ask about how to handle the problem of massive variation in some wards
        ward = 28
        correlation = np.zeros(110)
        for time in range(0, 110):
            priceData = bristolData.iloc[:, time]
            correlation[time] = priceData.corr(distanceData.iloc[ward, :], method='pearson')

        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        # ax[0] is correlation plotted over time
        ax[0].scatter(np.linspace(1995.75, 2023.25, 110), correlation,
                      label='Price Closeness Correlation', marker='x', color='teal')
        ax[0].axvline(2008, ls='--', color='orange')
        ax[0].axvline(2009, ls='--', color='orange', label='2008-2009 Financial Crisis')

        ax[0].set_xlabel('Year')
        ax[0].set_ylabel("Pearson's Correlation Coefficient")

        ax[0].legend()
        ax[0].grid()

        # ax[1] is plot of closeness-price
        ax1x = distanceData.iloc[ward, :]
        ax[1].scatter(ax1x.iloc[::-1], bristolData.iloc[:, -20], marker='x', color='teal')
        ax[1].grid()

        ax[1].set_xlabel('Closeness')
        ax[1].set_ylabel('Price (£)')
        print(correlation[0])
        print(correlation[-1])
        
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(__file__), 'Figures', 'Correlation_Intermediary.pdf'), format='pdf')


    # actual graph of ward correlations
    bristolData = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data',
                              'Wards', 'Median_Prices.csv'), index_col=0)

    # ward distances
    distanceData = pd.read_csv(os.path.join(os.path.dirname(__file__),
                               'distance_matrix.csv'), index_col=0)

    # normalising & inverting the distances
    distanceData = distanceData.map(lambda x: 1-((x-distanceData.min(axis=None))/(distanceData.max(axis=None)-distanceData.min(axis=None))))

    # enacting inverse square law
    # distanceData = distanceData.map(lambda x: x**2)

    # time series plot
    TimeSeriesPlotting(bristolData, distanceData)

    # find the correlation for each ward, take the average since 2010 & plot
    corrMean = np.zeros(34)
    corrSTD = np.zeros(34)
    corrMed = np.zeros(34)
    for ward in range(0, 34):
        correlation = np.zeros(110-57)
        for time in range(57, 110):
            priceData = bristolData.iloc[:, time]
            correlation[time-57] = priceData.corr(distanceData.iloc[ward, :], method='pearson')

        corrMean[ward] = correlation.mean()
        corrMed[ward] = np.median(correlation)
        corrSTD[ward] = correlation.std()

    # print(corrMed)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(list(bristolData.index.values), corrMean, color='teal')

    # ax.errorbar(list(bristolData.index.values), corrMean, yerr=corrSTD, color='black')

    ax.axhline(0.3487, color='red', ls='--', label=r'Critical value, $r=\pm 0.3487$')
    ax.axhline(-0.3487, color='red', ls='--')
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim([-0.4, 0.8])
    # ax.set_xlim(['Ashley', 'St George West'])
    fig.tight_layout()
    ax.legend()

    correlationData = pd.DataFrame(corrMean, index = list(bristolData.index.values))
    correlationData[:, 0] = pd.to_numeric(correlationData.iloc[:, 0])
    correlationData.to_csv('Distance_Correlation.csv', header=False)

    fig.savefig(os.path.join(os.path.dirname(__file__), 'Figures', 'Ward Distance Correlation Bar.pdf'), format='pdf')


def ARIMAX_Results_Plotting():
    # get most recent price data for sorting wards in plots
    prices = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'Median_Prices.csv'), index_col=0).iloc[:,-1]
    # distances = pd.read_csv(os.path.join(os.path.dirname))

    directory = os.path.join(os.path.dirname(__file__), 'Data', 'ARIMA_Result')
    # r2_arima = pd.read_csv(os.path.join(directory, 'ARIMA_r2.csv'), index_col=0)
    r2_results = pd.read_csv(os.path.join(directory, 'ARIMAX_r2_Kate.csv'), index_col=0)

    # r2_crisis_arima = pd.read_csv(os.path.join(directory, 'ARIMA_r2_crisis.csv'), index_col=0)
    # r2_crisis_results = pd.read_csv(os.path.join(directory, 'ARIMAX_r2_crisis.csv'), index_col=0)

    # p_arima = pd.read_csv(os.path.join(directory, 'ARIMA_p.csv'), index_col=0)
    # p_results = pd.read_csv(os.path.join(directory, 'ARIMAX_p.csv'), index_col=0)

    # q_arima = pd.read_csv(os.path.join(directory, 'ARIMA_q.csv'), index_col=0)
    # q_results = pd.read_csv(os.path.join(directory, 'ARIMAX_q.csv'), index_col=0)



    # r2_results.insert(0, '0', r2_arima['0'])
    # p_results.insert(0, '0', p_arima['0'])
    # q_results.insert(0, '0', q_arima['0'])

    # r2_crisis_results.insert(0, '0', r2_crisis_arima['0'])

    features = range(0, 16)

    Stoke_Bishop_Pred = False
    if Stoke_Bishop_Pred:
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        ax[0].plot(features, r2_results.loc['Stoke Bishop'], color='red', label='Stoke Bishop')
        ax[0].plot(features, r2_results.loc['Horfield'], color='teal', label='Horfield')
        ax[0].plot(features, r2_results.loc['Lawrence Hill'], color='green', label='Lawrence Hill')
        
        ax[0].grid()
        ax[0].set_xlabel('Exogenous Features (cumulative)')
        ax[0].set_ylabel(r'$R^2$')
        ax[0].legend(title='Ward:')
    
        # ax[1].plot(features, p_results.loc['Stoke Bishop'], color='red', ls='--')
        # ax[1].plot(features, q_results.loc['Stoke Bishop'], color='red', ls=':')
        # ax[1].plot(features, p_results.loc['Horfield'], color='teal', ls='--')
        # ax[1].plot(features, q_results.loc['Horfield'], color='teal', ls=':')
        # ax[1].plot(features, p_results.loc['Lawrence Hill'], color='green', ls='--')
        # ax[1].plot(features, q_results.loc['Lawrence Hill'], color='green', ls=':')
        # ax[1].legend([r'$p$', r'$q$'], title='Hyperparameter', loc=1)
    
        ax[1].grid()
        ax[1].set_xlabel('Exogenous Features (cumulative)')
    
        fig.savefig(os.path.join(os.path.dirname(__file__), 'Figures', 'Stoke_Bishop_Prediction.pdf'), format='pdf')

    
    All_Pred = True
    if All_Pred:
        fig, ax = plt.subplots(ncols=2, figsize=(12,6))
        r2_max_idx = r2_results.idxmax(axis=1)
        # r2_crisis_max_idx = r2_crisis_results.idxmax(axis=1)
        for ward in range(0, len(r2_max_idx.index.tolist())):
            ax[0].scatter(prices.iloc[ward], r2_results.iloc[ward, int(r2_max_idx.iloc[ward])], color='red')
            # ax[0].scatter(prices.iloc[ward], r2_crisis_results.iloc[ward, int(r2_crisis_max_idx.iloc[ward])], color='teal')
            # ax[1].scatter(prices.iloc[ward], p_results.iloc[ward, int(r2_max_idx.iloc[ward])], color='red', label=r'$p$')
            # ax[1].scatter(prices.iloc[ward], q_results.iloc[ward, int(r2_max_idx.iloc[ward])], color='teal', label=r'$q$')

        ax[0].grid()
        ax[0].set_xlabel('Price (£)')
        ax[0].set_ylabel(r'$R^2$')
        
        ax[1].grid()
        ax[1].legend(['p', 'q'], title='Hyperparameters:')
        ax[1].set_xlabel('Price (£)')


def main():
    ARIMAX_Results_Plotting()
    # DistanceCorrelation2()
    # DistanceCorrelation()
    # NetDwellings()
    # HouseTypeSales()
    # MedianWardPrices()
    # EnglandWalesMedianPrices()


if __name__ == '__main__':
    main()
    