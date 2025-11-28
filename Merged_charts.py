import json
import re
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import folium
from dash import Dash, dcc, html, Input, Output
geojson_path = "ph_regions.geojson"
bank_csv = "banks_long_format_clean_Nov21.csv"
atm_csv = "atm_long_format_clean_Nov21.csv"
gdp_csv = "GDP_Quarterly_CubicSpline_2020_2025.csv"
cpi_csv = "CPI_Quarterly_CubicSpline_2020_2025.csv"
pop_csv = "Population_Remapped_Quarterly_2020_2025.csv"

age_filter = "Legal Ages (10+)"
max_quarter = "2025Q2"

map_height_px = 1075  # single source of truth for height


def norm_letters(s: str) -> str:
    return re.sub(r"[^A-Za-z]+", "", str(s))


def quarter_sort_key(q: str):
    q = str(q)
    year = int(q[:4])
    qn = int(q[-1]) if "Q" in q else 1
    return year, qn


def minmax_scale(group):
    vmin = group["value"].min()
    vmax = group["value"].max()
    if vmax == vmin:
        group["value_scaled"] = 0.5
    else:
        group["value_scaled"] = (group["value"] - vmin) / (vmax - vmin)
    return group


def quarter_label(q: str) -> str:
    return f"{q[2:4]}Q{q[-1]}"


prov_to_region = {
    #"MetropolitanManila": "National Capital Region",
    "Abra": "Cordillera Administrative Region (CAR)",
    "Apayao": "Cordillera Administrative Region (CAR)",
    "Benguet": "Cordillera Administrative Region (CAR)",
    "Ifugao": "Cordillera Administrative Region (CAR)",
    "Kalinga": "Cordillera Administrative Region (CAR)",
    "MountainProvince": "Cordillera Administrative Region (CAR)",
    "Basilan": "Autonomous Region in Muslim Mindanao (ARMM)",
    "LanaodelSur": "Autonomous Region in Muslim Mindanao (ARMM)",
    "Maguindanao": "Autonomous Region in Muslim Mindanao (ARMM)",
    "Sulu": "Autonomous Region in Muslim Mindanao (ARMM)",
    "TawiTawi": "Autonomous Region in Muslim Mindanao (ARMM)",
    "IlocosNorte": "Region I (Ilocos Region)",
    "IlocosSur": "Region I (Ilocos Region)",
    "LaUnion": "Region I (Ilocos Region)",
    "Pangasinan": "Region I (Ilocos Region)",
    "Batanes": "Region II (Cagayan Valley)",
    "Cagayan": "Region II (Cagayan Valley)",
    "Isabela": "Region II (Cagayan Valley)",
    "NuevaVizcaya": "Region II (Cagayan Valley)",
    "Quirino": "Region II (Cagayan Valley)",
    "Aurora": "Region III (Central Luzon)",
    "Bataan": "Region III (Central Luzon)",
    "Bulacan": "Region III (Central Luzon)",
    "NuevaEcija": "Region III (Central Luzon)",
    "Pampanga": "Region III (Central Luzon)",
    "Tarlac": "Region III (Central Luzon)",
    "Zambales": "Region III (Central Luzon)",
    "Batangas": "Region IV-A (CALABARZON)",
    "Cavite": "Region IV-A (CALABARZON)",
    "Laguna": "Region IV-A (CALABARZON)",
    "Quezon": "Region IV-A (CALABARZON)",
    "Rizal": "Region IV-A (CALABARZON)",
    "Marinduque": "Region IV-B (MIMAROPA)",
    "OccidentalMindoro": "Region IV-B (MIMAROPA)",
    "OrientalMindoro": "Region IV-B (MIMAROPA)",
    "Palawan": "Region IV-B (MIMAROPA)",
    "Romblon": "Region IV-B (MIMAROPA)",
    "Albay": "Region V (Bicol Region)",
    "CamarinesNorte": "Region V (Bicol Region)",
    "CamarinesSur": "Region V (Bicol Region)",
    "Catanduanes": "Region V (Bicol Region)",
    "Masbate": "Region V (Bicol Region)",
    "Sorsogon": "Region V (Bicol Region)",
    "Aklan": "Region VI (Western Visayas)",
    "Antique": "Region VI (Western Visayas)",
    "Capiz": "Region VI (Western Visayas)",
    "Guimaras": "Region VI (Western Visayas)",
    "Iloilo": "Region VI (Western Visayas)",
    "NegrosOccidental": "Region VI (Western Visayas)",
    "Bohol": "Region VII (Central Visayas)",
    "Cebu": "Region VII (Central Visayas)",
    "NegrosOriental": "Region VII (Central Visayas)",
    "Siquijor": "Region VII (Central Visayas)",
    "Biliran": "Region VIII (Eastern Visayas)",
    "EasternSamar": "Region VIII (Eastern Visayas)",
    "Leyte": "Region VIII (Eastern Visayas)",
    "NorthernSamar": "Region VIII (Eastern Visayas)",
    "Samar": "Region VIII (Eastern Visayas)",
    "SouthernLeyte": "Region VIII (Eastern Visayas)",
    "ZamboangadelNorte": "Region IX (Zamboanga Peninsula)",
    "ZamboangadelSur": "Region IX (Zamboanga Peninsula)",
    "ZamboangaSibugay": "Region IX (Zamboanga Peninsula)",
    "Bukidnon": "Region X (Northern Mindanao)",
    "Camiguin": "Region X (Northern Mindanao)",
    "LanaodelNorte": "Region X (Northern Mindanao)",
    "MisamisOccidental": "Region X (Northern Mindanao)",
    "MisamisOriental": "Region X (Northern Mindanao)",
    "DavaoOriental": "Region XI (Davao Region)",
    "DavaodelNorte": "Region XI (Davao Region)",
    "DavaodelSur": "Region XI (Davao Region)",
    "DavaoOccidental": "Region XI (Davao Region)",
    "CompostelaValley": "Region XI (Davao Region)",
    "NorthCotabato": "Region XII (SOCCSKSARGEN)",
    "SouthCotabato": "Region XII (SOCCSKSARGEN)",
    "SultanKudarat": "Region XII (SOCCSKSARGEN)",
    "Sarangani": "Region XII (SOCCSKSARGEN)",
    "AgusandelNorte": "Region XIII (Caraga)",
    "AgusandelSur": "Region XIII (Caraga)",
    "DinagatIslands": "Region XIII (Caraga)",
    "SurigaodelNorte": "Region XIII (Caraga)",
    "SurigaodelSur": "Region XIII (Caraga)",
}

with open(geojson_path, "r", encoding="utf-8") as f:
    gj = json.load(f)

gdf = gpd.GeoDataFrame.from_features(gj["features"])
gdf.set_crs(epsg=4326, inplace=True)

prov_col = "NAME_1" if "NAME_1" in gdf.columns else gdf.columns[0]
gdf["name_1_token"] = gdf[prov_col].apply(norm_letters)

df_geo_map = pd.DataFrame(
    {"name_1_token": list(prov_to_region.keys()), "region": list(prov_to_region.values())}
)

gdf = gdf.merge(df_geo_map, on="name_1_token", how="left")
gdf = gdf.dropna(subset=["region"]).copy()

region_gdf = gdf.dissolve(by="region", aggfunc="first").reset_index()
region_gdf.set_crs(epsg=4326, inplace=True)
merged_geojson = json.loads(region_gdf.to_json())
regions_geo = set(region_gdf["region"].unique())

# use a manual bounding box that safely covers the PH
map_bounds = [[4.0, 116.0], [21.5, 127.0]]

# ------------- data prep (same as before, omitted comments) -------------

banks_raw = pd.read_csv(bank_csv)
banks_raw["date_parsed"] = pd.to_datetime(
    banks_raw["Date"], format="%d/%m/%Y", errors="coerce"
)
banks_raw["quarter"] = banks_raw["date_parsed"].dt.to_period("Q").astype(str)

banks_tot = (
    banks_raw[banks_raw["Type"] == "Total"]
    .groupby(["Region", "quarter"], as_index=False)["Total"]
    .sum()
)
banks_tot.rename(columns={"Region": "region", "Total": "value"}, inplace=True)
banks_tot["metric_type"] = "Bank Offices"

atm_raw = pd.read_csv(atm_csv)
atm_raw["date_parsed"] = pd.to_datetime(
    atm_raw["Date"], format="%d/%m/%Y", errors="coerce"
)
atm_raw["quarter"] = atm_raw["date_parsed"].dt.to_period("Q").astype(str)

atm_tot = (
    atm_raw[atm_raw["Type"] == "Total"]
    .groupby(["Region", "quarter"], as_index=False)["Total"]
    .sum()
)
atm_tot.rename(columns={"Region": "region", "Total": "value"}, inplace=True)
atm_tot["metric_type"] = "ATMs"

atm_regions = set(atm_tot["region"].unique())
missing_regions = regions_geo - atm_regions
if missing_regions:
    quarters_atm = sorted(atm_tot["quarter"].unique(), key=quarter_sort_key)
    rows = []
    for r in missing_regions:
        for q in quarters_atm:
            rows.append({"region": r, "quarter": q, "value": 0, "metric_type": "ATMs"})
    atm_tot = pd.concat([atm_tot, pd.DataFrame(rows)], ignore_index=True)

gdp_df = pd.read_csv(gdp_csv)
gdp_all = gdp_df[gdp_df["Type_1"] == "All"].copy()
gdp_all = gdp_all[
    ~gdp_all["Location_Renamed"].isin(["Philippines", "Negros Island Region (NIR)"])
]
gdp_rq = (
    gdp_all.groupby(["Location_Renamed", "Quarter"], as_index=False)["GDP_Quarterly"]
    .sum()
)
gdp_rq.rename(
    columns={"Location_Renamed": "region", "Quarter": "quarter", "GDP_Quarterly": "value"},
    inplace=True,
)
gdp_rq["metric_type"] = "GDP"

cpi_df = pd.read_csv(cpi_csv)
cpi_all = cpi_df[
    (cpi_df["Type_1"] == "0 - ALL ITEMS")
    & (cpi_df["Type_2"] == "0 - ALL ITEMS")
    & (cpi_df["Type_3"] == "0 - ALL ITEMS")
    & (cpi_df["Type_4"] == "0 - ALL ITEMS")
].copy()
cpi_all = cpi_all[
    ~cpi_all["Location_Renamed"].isin(["PHILIPPINES", "Negros Island Region (NIR)"])
]
cpi_rq = (
    cpi_all.groupby(["Location_Renamed", "Quarter"], as_index=False)["CPI_Quarterly"]
    .mean()
)
cpi_rq.rename(
    columns={"Location_Renamed": "region", "Quarter": "quarter", "CPI_Quarterly": "value"},
    inplace=True,
)
cpi_rq["metric_type"] = "CPI"

pop_df = pd.read_csv(pop_csv)
pop_leg = pop_df[pop_df["age_group"] == age_filter].copy()
pop_leg = pop_leg[pop_leg["Region_Remap"] != "Philippines"]
pop_rq = (
    pop_leg.groupby(["Region_Remap", "Quarter"], as_index=False)[
        "population"
    ]
    .sum()
)
pop_rq.rename(
    columns={
        "Region_Remap": "region",
        "Quarter": "quarter",
        "population": "value",
    },
    inplace=True,
)
pop_rq["metric_type"] = "Population Density"

map_df = pd.concat([banks_tot, atm_tot, gdp_rq, cpi_rq, pop_rq], ignore_index=True)
map_df = map_df[map_df["region"].isin(regions_geo)].copy()
map_df["quarter"] = map_df["quarter"].astype(str)
map_df["region"] = map_df["region"].astype(str)

max_key = quarter_sort_key(max_quarter)
map_df = map_df[map_df["quarter"].apply(lambda x: quarter_sort_key(x) <= max_key)]

#print(map_df)

metrics_list = ["Bank Offices", "ATMs", "GDP", "CPI", "Population Density"]
metrics_list = [m for m in metrics_list if m in map_df["metric_type"].unique()]

metric_colors = {
    "Bank Offices": "Greens",
    "ATMs": "Blues",
    "GDP": "Purples",
    "CPI": "Oranges",
    "Population Density": "Reds",
}

metric_vmin = map_df.groupby("metric_type")["value"].min().to_dict()
metric_vmax = map_df.groupby("metric_type")["value"].max().to_dict()
#print(map_df.groupby("metric_type")["value"].min().to_dict())
#print(map_df.groupby("metric_type")["value"].max().to_dict())
# bar / heatmap prep same as before (unchanged)
df_banks = pd.read_csv(bank_csv)
df_banks["Date"] = pd.to_datetime(
    df_banks["Date"], format="%d/%m/%Y", errors="coerce"
)
df_banks["quarter"] = df_banks["Date"].dt.to_period("Q").astype(str)
df_banks = df_banks.rename(columns={"Region": "region", "Total": "counts"})
df_banks = df_banks[df_banks["Type"] != "Total"].copy()
df_banks["metric"] = "Bank Offices"

df_atms = pd.read_csv(atm_csv)
df_atms["Date"] = pd.to_datetime(
    df_atms["Date"], format="%d/%m/%Y", errors="coerce"
)
df_atms["quarter"] = df_atms["Date"].dt.to_period("Q").astype(str)
df_atms = df_atms.rename(columns={"Region": "region", "Total": "counts"})
df_atms = df_atms[df_atms["Type"] != "Total"].copy()
df_atms["metric"] = "ATMs"

df_ba = pd.concat([df_banks, df_atms], ignore_index=True)
df_ba["quarter"] = df_ba["quarter"].astype(str).str.strip()
df_ba = df_ba[df_ba["quarter"].apply(lambda x: quarter_sort_key(x) <= max_key)]

regions_list = sorted(df_ba["region"].unique())

type_order = [
    "Universal and Commercial Banks",
    "Thrift Banks",
    "Rural and Cooperative Banks",
    "Digital Banks",
]

metric_color_bar = {
    "Bank Offices": "#4C72B0",
    "ATMs": "#C44E52",
}

gdp_long = gdp_rq[["region", "quarter", "value"]].copy()
gdp_long["metric"] = "GDP (current prices, PHP)"

cpi_long = cpi_rq[["region", "quarter", "value"]].copy()
cpi_long["metric"] = "CPI (2018=100)"

pop_long = pop_rq[["region", "quarter", "value"]].copy()
pop_long["metric"] = "Population (Legal Ages 10+)"

macro_long = pd.concat([gdp_long, cpi_long, pop_long], ignore_index=True)
macro_long["quarter"] = macro_long["quarter"].astype(str).str.strip()
macro_long["region"] = macro_long["region"].astype(str).str.strip()
macro_long = macro_long.groupby("metric", group_keys=False).apply(minmax_scale)
#print(macro_long.head())

regions_order = sorted(macro_long["region"].unique())
metrics_order_overall = [
    "GDP (current prices, PHP)",
    "CPI (2018=100)",
    "Population (Legal Ages 10+)",
]

gdp_min, gdp_max = gdp_rq["value"].min(), gdp_rq["value"].max()
cpi_min, cpi_max = cpi_rq["value"].min(), cpi_rq["value"].max()
pop_min, pop_max = pop_rq["value"].min(), pop_rq["value"].max()


def make_overall_heatmap(q):
    sub = macro_long[macro_long["quarter"] == q].copy()
    if sub.empty:
        fig = px.imshow([[0]], x=[""], y=[""])
        fig.update_layout(
            title=f"No macro data for {q}",
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    pivot = (
        sub.pivot(index="region", columns="metric", values="value_scaled")
        .reindex(index=regions_order, columns=metrics_order_overall)
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"x": "", "y": "Region", "color": "Scaled value (0–1)"},
        title=f"Macro indicators — {q}",
    )
    fig.update_layout(
        margin=dict(l=150, r=40, t=50, b=40),
        height=380,
        coloraxis_colorbar=dict(
            title="Scaled value (0–1)",
            ticks="outside",
            len=0.7,
            thickness=14,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    return fig


quarters_all = sorted(
    set(map_df["quarter"]).union(set(df_ba["quarter"])).union(set(macro_long["quarter"])),
    key=quarter_sort_key,
)

app = Dash()
app.title = "PH Regional Dashboard"

initial_quarter = quarters_all[0]
initial_overall_fig = make_overall_heatmap(initial_quarter)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "18px"},
    children=[
        html.H2(
            "Philippines Regional Dashboard: Banks, ATMs, GDP, CPI, Population",
            style={"marginBottom": "4px"},
        ),
        #html.P(
        #    "Use the controls to change the quarter, map metric, and bank-type mix.",
        #    style={"marginTop": "0px", "color": "#555", "marginBottom": "8px"},
        #),
        html.Div(
            style={"marginBottom": "12px"},
            children=[
                html.Label(
                    "Quarter (global time scale)",
                    style={"fontWeight": "bold", "fontSize": "15px"}, #edited from 13px
                ),
                dcc.Slider(
                    id="quarter_slider",
                    min=0,
                    max=len(quarters_all) - 1,
                    step=1,
                    value=0,
                    marks={
                        i: quarter_label(q)
                        for i, q in enumerate(quarters_all)
                        if (i == 0 or i == len(quarters_all) - 1 or q.endswith("Q1"))
                    },
                ),
                #html.Div(
                #    id="quarter_text",
                #    style={
                #        "fontSize": "15px", #edited from 12px
                #        "marginTop": "3px",
                #        "fontWeight": "bold",
                #        "color": "#333",
                #    },
                #),
            ],
        ),
        html.Div(
            style={
                "display": "flex",
                "gap": "14px",
                "alignItems": "center",
                "marginBottom": "10px",
            },
            children=[
                html.Div(
                    style={"minWidth": "260px"},
                    children=[
                        html.Label(
                            "Region for bank / ATM bar chart",
                            style={"fontWeight": "bold", "fontSize": "15px"}, #edited from 13px
                        ),
                        dcc.Dropdown(
                            id="region_dropdown",
                            options=[{"label": r, "value": r} for r in regions_list],
                            value=(
                                "National Capital Region"
                                if "National Capital Region" in regions_list
                                else regions_list[0]
                            ),
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "260px"},
                    children=[
                        html.Label(
                            "Metric for choropleth map",
                            style={"fontWeight": "bold", "fontSize": "15px"}, #edited from 13px
                        ),
                        dcc.RadioItems(
                            id="metric_radio",
                            options=[{"label": m, "value": m} for m in metrics_list],
                            value=metrics_list[0],
                            labelStyle={
                                "display": "inline-block",
                                "marginRight": "10px",
                            },
                            style={"fontSize": "15px"}, #edited from 13px
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "18px", "alignItems": "flex-start"},
            children=[
                # left column unchanged (bank bar + heatmaps) ...
                html.Div(
                    style={
                        "flex": "0 0 45%",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "14px",
                    },
                    children=[
                        html.Div(
                            style={
                                "border": "1px solid #ddd",
                                "borderRadius": "8px",
                                "padding": "10px",
                                "backgroundColor": "#fafafa",
                            },
                            children=[
                                html.H4(
                                    "Bank Type Mix by Region",
                                    style={"marginTop": 0, "marginBottom": "4px"},
                                ),
                                dcc.Graph(
                                    id="bank_mix_graph",
                                    style={"height": "420px"}, #edited from 320px
                                ),
                                html.Div(
                                    id="ratio_box",
                                    style={"marginTop": "6px", "fontSize": "15px"}, #edited from 12px
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "border": "1px solid #ddd",
                                "borderRadius": "8px",
                                "padding": "10px",
                                "backgroundColor": "#fafafa",
                            },
                            children=[
                                html.H4(
                                    "Macro Heatmaps by Region & Quarter",
                                    style={"marginTop": 0, "marginBottom": "4px"},
                                ),
                                dcc.Tabs(
                                    id="heatmap_tabs",
                                    value="tab_overall",
                                    children=[
                                        dcc.Tab(
                                            label="Overall (scaled)",
                                            value="tab_overall",
                                            children=[
                                                dcc.Graph(
                                                    id="overall_heatmap",
                                                    figure=initial_overall_fig,
                                                    style={"height": "420px"},
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="GDP",
                                            value="tab_gdp",
                                            children=[
                                                dcc.Graph(
                                                    id="gdp_heatmap",
                                                    style={"height": "420px"},
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="CPI",
                                            value="tab_cpi",
                                            children=[
                                                dcc.Graph(
                                                    id="cpi_heatmap",
                                                    style={"height": "420px"},
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Population",
                                            value="tab_pop",
                                            children=[
                                                dcc.Graph(
                                                    id="pop_heatmap",
                                                    style={"height": "420px"},
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                # right column map
                html.Div(
                    style={"flex": "0 0 55%"},
                    children=[
                        html.Div(
                            style={
                                "border": "1px solid #ddd",
                                "borderRadius": "8px",
                                "padding": "8px",
                                "backgroundColor": "#fafafa",
                            },
                            children=[
                                html.H4(
                                    "Choropleth Map",
                                    style={"marginTop": 0, "marginBottom": "4px"},
                                ),
                                html.Iframe(
                                    id="map_frame",
                                    style={
                                        "width": "100%",
                                        "height": f"{map_height_px}px",
                                        "border": "none",
                                    },
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)


#@app.callback(
#    Output("quarter_text", "children"),
#    Input("quarter_slider", "value"),
#)
#def update_quarter_text(idx):
#    q = quarters_all[idx]
#    return f"Selected quarter: {q}"


@app.callback(
    Output("map_frame", "srcDoc"),
    Input("metric_radio", "value"),
    Input("quarter_slider", "value"),
    #Input("region_dropdown", "value"),
)
def update_map(metric_sel, quarter_index): #, region_sel):
    q = quarters_all[quarter_index]
    sub = map_df[
        (map_df["metric_type"] == metric_sel) & (map_df["quarter"] == q) #& (map_df["region"] == region_sel)
    ].copy()

    fmap = folium.Map(
        location=[12.5, 122.5],
        zoom_start=5.2,
        tiles="cartodbpositron",
        width="100%",
        height=f"{map_height_px}px",
    )

    fmap.fit_bounds(map_bounds)

    if sub.empty:
        folium.Marker(
            [12.5, 122.5],
            popup=f"No data for {metric_sel} — {q}",
        ).add_to(fmap)
        return fmap.get_root().render()

    merged = region_gdf.merge(sub[["region", "value"]], on="region", how="left")

    vals = merged["value"].astype(float)

    #gvmin = metric_vmin.get(metric_sel)
    #gvmax = metric_vmax.get(metric_sel)
    #map_df.groupby("metric_type")["value"].min().to_dict()

    gvmin = sub["value"].min()
    gvmax = sub["value"].max()
    #print(gvmax)
    #print(q)
    if gvmin is None or gvmax is None:
        vmin = float(vals.min(skipna=True))
        vmax = float(vals.max(skipna=True))
    else:
        vmin = float(gvmin)
        vmax = float(gvmax)

    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    bins = list(np.linspace(vmin, vmax, 9))
    bins[0] -= 1e-9
    bins[-1] += 1e-9

    choropleth = folium.Choropleth(
        geo_data=merged,
        data=merged,
        columns=["region", "value"],
        key_on="feature.properties.region",
        fill_color=metric_colors.get(metric_sel, "Greens"),
        fill_opacity=0.8,
        line_opacity=0.9,
        line_color="black",
        legend_name=f"{metric_sel} — {q}",
        bins=bins,
        nan_fill_color="lightgrey",
        highlight=True,
    ).add_to(fmap)

    folium.GeoJsonTooltip(
        fields=["region", "value"],
        aliases=["Region:", "Value:"],
        localize=True,
        sticky=True,
        style=("background-color: white; color: #333333; font-family: arial; font-size: 25px; padding: 10px;")
    ).add_to(choropleth.geojson)

    return fmap.get_root().render()


@app.callback(
    Output("bank_mix_graph", "figure"),
    Output("ratio_box", "children"),
    Input("region_dropdown", "value"),
    Input("quarter_slider", "value"),
)
def update_bank_mix(region_sel, quarter_index):
    q = quarters_all[quarter_index]
    sub = df_ba[(df_ba["region"] == region_sel) & (df_ba["quarter"] == q)].copy()

    if sub.empty:
        fig = px.bar()
        fig.update_layout(
            title=f"No data for {region_sel} — {q}",
            xaxis_title="",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        kpi_children = [
            html.Div(
                f"No data for {region_sel} — {q}",
                style={"fontStyle": "italic", "color": "#777"},
            )
        ]
        return fig, kpi_children

    sub_type_metric = sub.groupby(["Type", "metric"], as_index=False)["counts"].sum()
    sub_type_metric["Type"] = pd.Categorical(
        sub_type_metric["Type"], categories=type_order, ordered=True
    )
    sub_type_metric = sub_type_metric.sort_values("Type")

    fig = px.bar(
        sub_type_metric,
        x="counts",
        y="Type",
        orientation="h",
        color="metric",
        barmode="group",
        color_discrete_map=metric_color_bar,
        title=f"{region_sel} — {q}",
        category_orders={"Type": type_order},
        labels={"counts": "Number of points of presence", "Type": "Bank type"},
    )
    fig.update_layout(
        legend_title_text="Channel",
        font=dict(family="Arial", size=20), #edited from 13
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=130, r=30, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    pivot = (
        sub.groupby(["Type", "metric"])["counts"]
        .sum()
        .unstack(fill_value=0)
    )

    cards = [
        html.Div(
            "Channel mix by bank type",
            style={
                "fontWeight": "bold",
                "fontSize": "12px",
                "marginBottom": "4px",
            },
        )
    ]

    for t in type_order:
        if t not in pivot.index:
            continue
        offices = int(pivot.loc[t].get("Bank Offices", 0))
        atms = int(pivot.loc[t].get("ATMs", 0))
        ratio = atms / offices if offices > 0 else None


        cards.append(
            html.Td(
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "6px",
                    "padding": "5px 7px",
                    "marginBottom": "4px",
                    "backgroundColor": "#ffffff",
                },
                children=[
                    html.Div(
                        t,
                        style={
                            "fontWeight": "bold",
                            "fontSize": "15px", #edited from 11px
                            "marginBottom": "1px",
                        },
                    ),
                    html.Div(f"Bank offices: {offices:,.0f}"),
                    html.Div(f"ATMs: {atms:,.0f}"),
                    html.Div(
                        "ATMs per bank office: "
                        + (f"{ratio:.2f}" if ratio is not None else "n/a"),
                        style={"marginTop": "1px", "fontWeight": "bold"},
                    ),
                ],
            )
        )

    return fig, cards


@app.callback(
    Output("overall_heatmap", "figure"),
    Output("gdp_heatmap", "figure"),
    Output("cpi_heatmap", "figure"),
    Output("pop_heatmap", "figure"),
    Input("quarter_slider", "value"),
)
def update_heatmaps(quarter_index):
    q = quarters_all[quarter_index]
    overall_fig = make_overall_heatmap(q)

    gdp_sub = gdp_rq[gdp_rq["quarter"] == q].copy()
    gdp_pivot = gdp_sub.pivot(index="region", columns="quarter", values="value")
    gdp_fig = px.imshow(
        gdp_pivot,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels={"x": "Quarter", "y": "Region", "color": "GDP"},
        title=f"GDP by region — {q}",
    )
    gdp_fig.update_layout(
        margin=dict(l=150, r=40, t=50, b=40),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    gdp_fig.update_xaxes(side="top")
    gdp_fig.update_coloraxes(cmin=gdp_min, cmax=gdp_max)

    cpi_sub = cpi_rq[cpi_rq["quarter"] == q].copy()
    cpi_pivot = cpi_sub.pivot(index="region", columns="quarter", values="value")
    cpi_fig = px.imshow(
        cpi_pivot,
        aspect="auto",
        color_continuous_scale="OrRd",
        labels={"x": "Quarter", "y": "Region", "color": "CPI"},
        title=f"CPI (all items) — {q}",
    )
    cpi_fig.update_layout(
        margin=dict(l=150, r=40, t=50, b=40),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    cpi_fig.update_xaxes(side="top")
    cpi_fig.update_coloraxes(cmin=cpi_min, cmax=cpi_max)

    pop_sub = pop_rq[pop_rq["quarter"] == q].copy()
    pop_pivot = pop_sub.pivot(index="region", columns="quarter", values="value")
    pop_fig = px.imshow(
        pop_pivot,
        aspect="auto",
        color_continuous_scale="Greens",
        labels={"x": "Quarter", "y": "Region", "color": "Population density"},
        title=f"Population (legal ages 10+) — {q}",
    )
    pop_fig.update_layout(
        margin=dict(l=150, r=40, t=50, b=40),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    pop_fig.update_xaxes(side="top")
    pop_fig.update_coloraxes(cmin=pop_min, cmax=pop_max)

    return overall_fig, gdp_fig, cpi_fig, pop_fig


#app.save("Philippines Regional Dashboard.html")
#app.run(debug=False, port=8060)
app.run(debug=False, host= '127.0.0.1', port=8085)
#http://127.0.0.1:8050