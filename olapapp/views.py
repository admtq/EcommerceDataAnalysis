from django.shortcuts import render
from django.shortcuts import redirect
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

def redirect_to_chart(request):
    return redirect('/olap-chart/?mode=1')


def olap_chart_view(request):
    mode = request.GET.get('mode', 'review')  # default ke 'review'

    df = pd.read_csv('OLAP_data.csv')

    if mode == '1':
        # ===========================
        # MODEL 1: Forecast Penjualan
        # ===========================

        # Transformasi wide → long
        df_long = pd.melt(
            df,
            id_vars=['product_name'],
            value_vars=[f'sales_month_{i}' for i in range(1, 13)],
            var_name='month',
            value_name='sales'
        )
        df_long['month'] = df_long['month'].str.extract('(\d+)').astype(int)
        df_long['order_month'] = pd.to_datetime('2023-' + df_long['month'].astype(str) + '-01')

        # Ambil top 5 produk
        total_sales = df_long.groupby('product_name')['sales'].sum().reset_index()
        top_5_products = total_sales.sort_values(by='sales', ascending=False).head(5)['product_name']
        df_top5 = df_long[df_long['product_name'].isin(top_5_products)]

        fig = go.Figure()

        for product in top_5_products:
            product_data = df_top5[df_top5['product_name'] == product].copy()
            product_data = product_data.sort_values('month')

            # === 1. Plot data aktual (bulan 1–12) ===
            fig.add_trace(go.Scatter(
                x=product_data['order_month'],
                y=product_data['sales'],
                mode='lines+markers',
                name=f'{product} (actual)',
                line=dict(dash='solid')
            ))

            # === 2. Regresi dan forecast (bulan 13–15) ===
            X = product_data[['month']]
            y = product_data['sales']
            model = LinearRegression()
            model.fit(X, y)

            future_months = np.arange(13, 16).reshape(-1, 1)
            future_sales = model.predict(future_months)
            future_dates = pd.date_range(start='2024-01-01', periods=3, freq='MS')

            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_sales,
                mode='lines+markers',
                name=f'{product} (forecast)',
                line=dict(dash='dash')
            ))

        fig.update_layout(
            title='Top 5 Product Sales per Month + 3-Month Forecast (Linear Regression)',
            xaxis_title='Bulan',
            yaxis_title='Penjualan'
        )

        plot_div = plot(fig, output_type='div')
        return render(request, 'olap_chart1.html', context={'plot_div': plot_div})


    elif mode == '2':
        # ===============================
        # MODEL 2: Clustering Review Score
        # ===============================

        df['review_score'] = df['review_score'].astype(str).str.replace(',', '.').astype(float)
        df_cluster = df.groupby('product_name')['review_score'].mean().reset_index()

        kmeans = KMeans(n_clusters=3, random_state=42)
        df_cluster['cluster'] = kmeans.fit_predict(df_cluster[['review_score']])

        fig = px.scatter(
            df_cluster,
            x='product_name',
            y='review_score',
            color='cluster',
            title='Clustering Produk Berdasarkan Review Score',
            labels={'product_name': 'Produk', 'review_score': 'Rata-Rata Skor'}
        )
        fig.update_layout(xaxis_tickangle=-45)

        plot_div = plot(fig, output_type='div')
        return render(request, 'olap_chart2.html', {'plot_div': plot_div})

    else:
        # ===============================
        # MODEL 3: Clustering total penjualan per kategori
        # ===============================

        # Total penjualan per kategori
        category_sales = df.groupby('category')['Total Sales'].sum().reset_index()

        # Clustering (misalnya: 3 cluster)
        kmeans = KMeans(n_clusters=3, random_state=42)
        category_sales['cluster'] = kmeans.fit_predict(category_sales[['Total Sales']])

        # Visualisasi
        fig = px.bar(
            category_sales,
            x='category',
            y='Total Sales',
            color='cluster',
            title='Clustering Total Penjualan per Kategori',
            labels={'Total Sales': 'Total Penjualan', 'category': 'Kategori'}
        )
        fig.show()

        plot_div = plot(fig, output_type='div')
        return render(request, 'olap_chart3.html', {'plot_div': plot_div})