# --- Import required libraries ---
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, Div, WMTSTileSource, ColorBar, BasicTicker
from bokeh.layouts import column, row
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
from pyproj import Transformer
import os
from datetime import datetime

# --- Gestion des erreurs pour vérifier les fichiers CSV ---
if not os.path.exists('sales_data.csv'):
    raise FileNotFoundError("Le fichier 'sales_data.csv' est manquant.")
if not os.path.exists('geographic_data.csv'):
    raise FileNotFoundError("Le fichier 'geographic_data.csv' est manquant.")
if not os.path.exists('customer_feedback.csv'):
    raise FileNotFoundError("Le fichier 'customer_feedback.csv' est manquant.")

# Charger les données des fichiers CSV
sales_data = pd.read_csv('sales_data.csv')
geo_data = pd.read_csv('geographic_data.csv')
customer_feedback = pd.read_csv('customer_feedback.csv')

# Vérification des colonnes nécessaires
required_columns_sales = {'date', 'category', 'sales'}
if not required_columns_sales.issubset(sales_data.columns):
    raise ValueError(f"Le fichier 'sales_data.csv' doit contenir les colonnes : {required_columns_sales}")

required_columns_geo = {'latitude', 'longitude', 'region', 'sales'}
if not required_columns_geo.issubset(geo_data.columns):
    raise ValueError(f"Le fichier 'geographic_data.csv' doit contenir les colonnes : {required_columns_geo}")

required_columns_feedback = {'date', 'category', 'rating', 'sentiment_score'}
if not required_columns_feedback.issubset(customer_feedback.columns):
    raise ValueError(f"Le fichier 'customer_feedback.csv' doit contenir les colonnes : {required_columns_feedback}")

# Préparation des données
sales_data['date'] = pd.to_datetime(sales_data['date'])
sales_data['day_of_week'] = sales_data['date'].dt.day_name()
customer_feedback['date'] = pd.to_datetime(customer_feedback['date'])

transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
geo_data['mercator_x'], geo_data['mercator_y'] = zip(*geo_data.apply(
    lambda row: transformer.transform(row['longitude'], row['latitude']),
    axis=1
))

# Fonction personnalisée pour remplacer CARTODBPOSITRON
def get_cartodbpositron():
    return WMTSTileSource(
        url="https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attribution="© OpenStreetMap contributors",
        tile_size=256
    )

# --- Partie 1 : Tendance des ventes quotidiennes ---
daily_sales = sales_data.groupby('date')['sales'].sum().reset_index()
line_source = ColumnDataSource(daily_sales)

p_line = figure(
    title="Tendance des Ventes Quotidiennes",
    x_axis_type='datetime', height=300, width=600,
    x_axis_label="Date", y_axis_label="Ventes (€)"
)
p_line.line('date', 'sales', source=line_source, line_width=2, color='blue')
p_line.add_tools(HoverTool(
    tooltips=[("Date", "@date{%F}"), ("Ventes (€)", "@sales{€0,0.00}")],
    formatters={'@date': 'datetime'}
))

# --- Partie 2 : Graphique en Barres pour Ventes par Catégorie ---
category_sales = sales_data.groupby('category')['sales'].sum().reset_index()
source_category = ColumnDataSource(category_sales)

p_bar = figure(
    x_range=category_sales['category'],
    title="Ventes Totales par Catégorie",
    height=300, width=600,
    x_axis_label="Catégories",
    y_axis_label="Ventes (€)"
)
p_bar.vbar(
    x='category', top='sales', source=source_category, width=0.7, color="orange"
)
p_bar.add_tools(HoverTool(tooltips=[("Catégorie", "@category"), ("Ventes (€)", "@sales{€0,0.00}")]))

# --- Partie 3 : Carte de Chaleur avec ColorBar ---
heatmap_data = sales_data.groupby(['day_of_week', 'category'])['sales'].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='category', values='sales').fillna(0)

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_pivot = heatmap_pivot.reindex(days_order)

heatmap_source = ColumnDataSource(heatmap_pivot.reset_index().melt(
    id_vars='day_of_week', var_name='category', value_name='sales'
))

p_heatmap = figure(
    title="Carte de Chaleur : Performance des Ventes",
    x_range=list(heatmap_pivot.columns), y_range=list(reversed(heatmap_pivot.index)),
    height=300, width=600, tools="hover",
    tooltips=[("Jour", "@day_of_week"), ("Catégorie", "@category"), ("Ventes (€)", "@sales{€0,0}")]
)
mapper = linear_cmap(
    field_name='sales', palette=Viridis256,
    low=heatmap_source.data['sales'].min(), high=heatmap_source.data['sales'].max()
)
p_heatmap.rect(x="category", y="day_of_week", width=1, height=1, source=heatmap_source, fill_color=mapper, line_color=None)

color_bar = ColorBar(
    color_mapper=mapper['transform'], width=8, location=(0, 0),
    ticker=BasicTicker(desired_num_ticks=10), title="Ventes (€)"
)
p_heatmap.add_layout(color_bar, 'right')

# --- Partie 4 : Carte Géographique ---
geo_source = ColumnDataSource(geo_data)
p_map = figure(
    title="Carte Géographique : Ventes par Région",
    x_axis_type="mercator", y_axis_type="mercator",
    width=800, height=400, tools="pan,wheel_zoom,box_zoom,reset,save"
)
tile_provider = get_cartodbpositron()
p_map.add_tile(tile_provider)
p_map.scatter(
    x='mercator_x', y='mercator_y', size=15, source=geo_source,
    fill_color="blue", fill_alpha=0.7, line_color=None, legend_label="Régions"
)
p_map.add_tools(HoverTool(
    tooltips=[("Région", "@region"), ("Ventes (€)", "@sales{0,0}"),
              ("Latitude", "@latitude"), ("Longitude", "@longitude")]
))

# --- Partie 5 : Analyse des Sentiments ---
def categorize_sentiment(score):
    if score > 0.7:
        return "Positive"
    elif 0.3 <= score <= 0.7:
        return "Neutral"
    else:
        return "Negative"

customer_feedback['Sentiment'] = customer_feedback['sentiment_score'].apply(categorize_sentiment)
sentiment_counts = customer_feedback['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
sentiment_source = ColumnDataSource(sentiment_counts)

p_sentiment = figure(
    title="Distribution des Sentiments Clients",
    x_range=sentiment_counts['Sentiment'], height=300, width=600,
    x_axis_label="Sentiment", y_axis_label="Nombre de Clients"
)
p_sentiment.vbar(x='Sentiment', top='Count', source=sentiment_source, width=0.8, color="green")
p_sentiment.add_tools(HoverTool(tooltips=[("Sentiment", "@Sentiment"), ("Nombre", "@Count")]))


# --- Ajout d'une date de dernière mise à jour ---
last_update = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
last_update_div = Div(text=f"<p>Dernière mise à jour : {last_update}</p>", width=400)

# --- Intégration dans un Tableau de Bord ---
dashboard_description = Div(text="""<h1>Tableau de Bord Interactif des Ventes</h1>
    <p>Bienvenue sur votre tableau de bord interactif !</p>
    <ul>
        <li><strong>Tendance des Ventes :</strong> Suivez l'évolution quotidienne des ventes.</li>
        <li><strong>Carte de Chaleur :</strong> Analysez les ventes par catégorie et jour de la semaine.</li>
        <li><strong>Carte Géographique :</strong> Explorez la distribution des ventes par région.</li>
        <li><strong>Feedback Clients :</strong> Visualisez les sentiments des clients.</li>
        <li><strong>Ventes par Catégorie :</strong> Consultez les ventes totales par catégorie.</li>
    </ul>
""", width=800)

charts_row = row(p_line, p_bar)
dashboard_layout = column(
    dashboard_description,
    last_update_div,
    charts_row,
    p_heatmap,
    p_map,
    p_sentiment
)

output_file("dashboard_interactif_final.html")
show(dashboard_layout)
