import pulp
import matplotlib.pyplot as plt
import xlsxwriter
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st



def define_project_parameters():
    """Définit les paramètres principaux du projet et les caractéristiques des engins."""
    project = {
        "total_volume": 2000000,  # Volume total à déplacer en m³
        "project_duration": 20,  # Durée totale du projet en mois
        "trips_per_minute": 2,  # Tours/minute
        "working_days_per_month": 20,  # Jours ouvrés/mois
        "wait_time": 12,  # Temps d'attente par défaut
        "max_simultaneous_engines": 12,  # Nombre maximal d'engins actifs simultanément
        "max_engines_in_gantt": 22,  # Nombre maximal d'engins représentés sur le diagramme de Gantt
    }

    engines = {
        'D1.0064': {"capacity": 1.4, "max_usage": 11},
        'D1.0134': {"capacity": 1.6, "max_usage": 11},
        'D1.0595': {"capacity": 1.75, "max_usage": 12},
        'D1.0596': {"capacity": 1.75, "max_usage": 16},
        'D1.0597': {"capacity": 1.75, "max_usage": 12},
        'D1.0598': {"capacity": 1.4, "max_usage": 13},
        'D1.0599': {"capacity": 1.4, "max_usage": 13},
        'D1.0504': {"capacity": 4.5, "max_usage": 13},
        'D1.0505': {"capacity": 4.5, "max_usage": 9},
        'D1.C001': {"capacity": 1.03, "max_usage": 11},
        'D1.C002': {"capacity": 1.03, "max_usage": 9},
        'D1.C009': {"capacity": 1.76, "max_usage": 13},
        'D1.C003': {"capacity": 2.5, "max_usage": 11},
        'D1.C005': {"capacity": 2.5, "max_usage": 7},
        'D1.C007': {"capacity": 2.5, "max_usage": 7},
        'D1.C008': {"capacity": 2.5, "max_usage": 7},
        'D1.C004': {"capacity": 3.3, "max_usage": 11},
        'D1.C006': {"capacity": 3.3, "max_usage": 15},
        'D1.C010': {"capacity": 5.1, "max_usage": 10},
        'D1.0544': {"capacity": 5.1, "max_usage": 10},
        'D1.0552': {"capacity": 5.1, "max_usage": 10},
         'D1.C013': {"capacity": 1.2, "max_usage": 20},
    }

    # Paramètre dynamique pour chaque engin : temps de travail quotidien

    for engine, data in engines.items():
        ci = data["capacity"]
        wait_time = project["wait_time"]  # Utilisation du paramètre du projet
        work_time = 7 * 60 * (1 - (wait_time / ((12 / ci) + wait_time)))
        data["work_time_per_day"] = work_time

    return project, engines

def generate_engine_period_combinations(engines):

    project , _ = define_project_parameters()
    """Crée une structure associant chaque engin à ses périodes d'utilisation en se basant sur max_usage."""
    engine_period_data = {}
    for engine, data in engines.items():
        # Utiliser max_usage pour définir la période d'utilisation
        for start_month in range(1, project["project_duration"] - data["max_usage"] + 2):
            end_month = start_month + data["max_usage"] - 1
            period_id = f"{engine}_{start_month}"  # ID unique pour chaque combinaison
            engine_period_data[period_id] = {
                "capacity": data["capacity"],
                "max_usage": data["max_usage"],
                "start_period": start_month,
                "end_period": end_month,
            }
    return engine_period_data




import pulp
import matplotlib.pyplot as plt
import xlsxwriter
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

# ... (other functions in pl_15_2_2.py)

def solve_optimization_problem(project, engine_period_data, engines):
    """Résout le problème d'optimisation pour minimiser le nombre d'engins utilisés."""
    months = list(range(1, project["project_duration"] + 1))
    model = pulp.LpProblem("Earthmoving_Optimization", pulp.LpMinimize)

    # Variables de décision
    engine_usage = pulp.LpVariable.dicts(
        "EngineUsage",
        ((engine, month) for engine in engine_period_data for month in range(
            engine_period_data[engine]["start_period"], engine_period_data[engine]["end_period"] + 1)),
        cat="Binary",
    )
    engine_active = pulp.LpVariable.dicts(
        "EngineActive", engine_period_data.keys(), cat="Binary"
    )

    # Dictionnaire pour suivre les engins de base utilisés
    base_engines_used = pulp.LpVariable.dicts(
        "BaseEngineUsed",
        {engine.split('_')[0] for engine in engine_period_data},  # Clés : noms d'engins de base
        cat="Binary"
    )

    # Fonction objectif : Minimiser le nombre total d'engins de base actifs
    model += pulp.lpSum([base_engines_used[engine] for engine in base_engines_used]), "Minimize_Total_Active_Engines"

    # Contraintes
    # Volume total à déplacer
    model += pulp.lpSum([
        engine_usage[(engine, month)] * engine_period_data[engine]["capacity"] *
        engines[engine.split('_')[0]]["work_time_per_day"] * project["trips_per_minute"] * project["working_days_per_month"]
        for engine in engine_period_data
        for month in range(engine_period_data[engine]["start_period"], engine_period_data[engine]["end_period"] + 1)
    ]) >= project["total_volume"], "Total_Volume"

    # Limitation des engins simultanés par mois
    for month in months:
        model += pulp.lpSum([
            engine_usage[(engine, month)] for engine in engine_period_data
            if engine_period_data[engine]["start_period"] <= month <= engine_period_data[engine]["end_period"]
        ]) <= project["max_simultaneous_engines"], f"Max_Simultaneous_Engines_Month_{month}"

    # Limitation des périodes d'utilisation par engin (basée sur max_usage)
    for engine in engine_period_data:
        model += pulp.lpSum([
            engine_usage[(engine, month)] for month in range(
                engine_period_data[engine]["start_period"], engine_period_data[engine]["end_period"] + 1)
        ]) <= engine_period_data[engine]["max_usage"], f"Max_Usage_Per_Engine_{engine}"

    # Contrainte additionnelle pour s'assurer que end_period - start_period <= max_usage
    for engine in engine_period_data:
        model += engine_period_data[engine]["end_period"] - engine_period_data[engine]["start_period"] + 1 <= engine_period_data[engine]["max_usage"], f"Duration_Limit_{engine}"

    # Liaison entre l'activité des engins et leur utilisation
    for engine in engine_period_data:
        for month in range(engine_period_data[engine]["start_period"], engine_period_data[engine]["end_period"] + 1):
            model += engine_active[engine] >= engine_usage[(engine, month)], f"Link_Active_And_Usage_{engine}_{month}"

    # Contrainte pour lier base_engines_used et engine_active
    for base_engine in base_engines_used:
        model += base_engines_used[base_engine] >= pulp.lpSum(
            [engine_active[engine] for engine in engine_period_data if engine.startswith(base_engine + "_")]
        ), f"Link_Base_Engine_Used_{base_engine}"

    # Limitation du nombre d'engins actifs au moins une fois
    model += pulp.lpSum([engine_active[engine] for engine in engine_period_data]) <= project["max_engines_in_gantt"], \
        "Max_Active_Engines"

    # Résolution
    model.solve()
    return model, engine_usage




def plot_gantt_chart(model, engine_usage, engine_period_data, project):
    """Affiche un diagramme de Gantt pour les engins utilisés."""
    if model.status != pulp.LpStatusOptimal:
        print("Solution non optimale ou problème infaisable.")
        return

    # Collecte des données pour les engins actifs
    active_engines = {}
    for (engine, month), var in engine_usage.items():
        if var.varValue == 1:
            if engine not in active_engines:
                active_engines[engine] = []
            active_engines[engine].append(month)

    # Création du diagramme de Gantt
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20.colors
    for idx, (engine, months) in enumerate(active_engines.items()):
        for month in months:
            plt.barh(engine.split('_')[0], width=1, left=month - 1, color=colors[idx % len(colors)], edgecolor="black")

    plt.xlabel("Mois")
    plt.ylabel("Engins")
    plt.title("Diagramme de Gantt : Utilisation des engins")
    plt.xticks(range(project["project_duration"]), [f"M{m}" for m in range(1, project["project_duration"] + 1)], rotation=45)
    plt.tight_layout()
    plt.show()



def generate_professional_excel(project, engines, active_engines, engine_period_data):
    """Génère un fichier Excel formaté avec les données du projet."""
    workbook = xlsxwriter.Workbook("Rapport_Optimisation.xlsx")
   

       # Définir les formats de style
    header_format = workbook.add_format({"bold": True, "bg_color": "#4F81BD", "font_color": "white", "align": "center", "border": 1})
    cell_format = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})
    total_format = workbook.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1, "align": "center"})

    # 1. Feuille des paramètres du projet
    worksheet_params = workbook.add_worksheet("Paramètres du Projet")
    worksheet_params.set_column("A:B", 40)

    worksheet_params.write("A1", "Paramètres", header_format)
    worksheet_params.write("B1", "Valeurs", header_format)

    project_params = [
        ("Volume total à déplacer (m³)", project["total_volume"]),
        ("Durée totale du projet (mois)", project["project_duration"]),
        ("Tours/minute", project["trips_per_minute"]),
        ("Jours ouvrés/mois", project["working_days_per_month"]),
        ("Nombre max d'engins simultanés", project["max_simultaneous_engines"]),
    ]

    for row, (param, value) in enumerate(project_params, start=1):
        worksheet_params.write(row, 0, param, cell_format)
        worksheet_params.write(row, 1, value, cell_format)

    # 2. Feuille des caractéristiques des engins
    worksheet_engines = workbook.add_worksheet("Caractéristiques des Engins")
    worksheet_engines.set_column("A:D", 20)

    headers = ["Code Engin", "Capacité (m³)", "Utilisation Max (mois)", "Temps de Travail Quotidien (minutes)"]
    for col, header in enumerate(headers):
        worksheet_engines.write(0, col, header, header_format)

    for row, (engine, data) in enumerate(engines.items(), start=1):
        worksheet_engines.write(row, 0, engine, cell_format)
        worksheet_engines.write(row, 1, data["capacity"], cell_format)
        worksheet_engines.write(row, 2, data["max_usage"], cell_format)
        worksheet_engines.write(row, 3, round(data["work_time_per_day"], 2), cell_format)  # Temps de travail quotidien

    # 3. Feuille des engins utilisés
    worksheet_used_engines = workbook.add_worksheet("Engins Utilisés")
    worksheet_used_engines.set_column("A:D", 30)
    worksheet_used_engines.set_column("E:E", 35)

    headers = ["Engin", "Capacité (m³)", "Quantité Transportée (m³)", "Nombre de Mois Utilisés"]
    for col, header in enumerate(headers):
        worksheet_used_engines.write(0, col, header, header_format)

    total_transport = 0
    row = 1
    for engine, months in active_engines.items():
        capacity = engine_period_data[engine]["capacity"]
        total_volume = (
            capacity
            * engines[engine.split('_')[0]]["work_time_per_day"]
            * project["trips_per_minute"]
            * project["working_days_per_month"]
            * len(months)
        )
        total_transport += total_volume
        worksheet_used_engines.write(row, 0, engine.split('_')[0], cell_format)
        worksheet_used_engines.write(row, 1, capacity, cell_format)
        worksheet_used_engines.write(row, 2, round(total_volume,2), cell_format)
        worksheet_used_engines.write(row, 3, len(months), cell_format)  # Nombre de mois utilisés
        row += 1

    # Ajouter la ligne de total
    worksheet_used_engines.merge_range(row, 0, row, 2, "Total Transporté (m³)", total_format)
    worksheet_used_engines.write(row, 3, round(total_transport,2), total_format)

    workbook.close()
    print("Rapport Excel formaté généré : Rapport_Optimisation.xlsx")

def extract_active_engines(engine_usage, engine_period_data):
    """Extrait les engins activement utilisés et les mois correspondants."""
    active_engines = {}
    for (engine, month), var in engine_usage.items():
        if var.varValue == 1:  # Si l'engin est utilisé dans ce mois
            if engine not in active_engines:
                active_engines[engine] = []
            active_engines[engine].append(month)
    return active_engines

def sensitivity_analysis_volume(model, engine_period_data, project, engines, parameter_name, variations):
    """Effectue une analyse de sensibilité pour un paramètre du projet."""
    results = []

    for variation in variations:
        modified_project = project.copy()
        modified_project[parameter_name] = project[parameter_name] + variation  # Modifier le volume total
        model, _ = solve_optimization_problem(modified_project, engine_period_data, engines)
        results.append((modified_project[parameter_name], pulp.value(model.objective)))

    # Visualisation des résultats
    plt.figure(figsize=(10, 6))
    param_values, objectives = zip(*results)  # Extraire les volumes réels et les objectifs
    plt.plot(param_values, objectives, marker='o', label=f'Impact de {parameter_name}')
    plt.xlabel('Volume total à déplacer (m³)')
    plt.ylabel('Nombre d\'engins nécessaires')
    plt.title('Analyse de sensibilité : Volume total de terre')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def sensitivity_analysis(model, engine_period_data, project, engines, parameter_name, variations):
    """Effectue une analyse de sensibilité pour un paramètre du projet."""
    results = []
    for variation in variations:
        modified_project = project.copy()
        modified_project[parameter_name] = 16 + variation  # Ajuster pour wait_time entre 8 et 16

        # Recalculer les temps de travail si le paramètre est `wait_time`
        if parameter_name == "wait_time":
            for engine, data in engines.items():
                ci = data["capacity"]
                wait_time = modified_project["wait_time"]
                data["work_time_per_day"] = 7 * 60 * (1 - (wait_time / ((12 / ci) + wait_time)))

        model, _ = solve_optimization_problem(modified_project, engine_period_data, engines)
        results.append((modified_project[parameter_name], pulp.value(model.objective)))

    # Visualisation des résultats
    plt.figure(figsize=(8, 6))
    variations, objectives = zip(*results)
    plt.plot(variations, objectives, marker='o', label=f'Impact de {parameter_name}')
    plt.xlabel('Valeur de wait_time (minutes)')
    plt.ylabel('Nombre d\'engins utilisés')
    plt.legend()
    plt.title('Analyse de sensibilité')
    plt.show()

def plot_sensitivity_analysis(results, param_name):
    """Affiche les résultats de l'analyse de sensibilité avec Plotly."""
    param_values, objectives = zip(*results)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=param_values, y=objectives, mode='lines+markers', name=f'Impact de {param_name}'))

    xlabel = 'Volume total à déplacer (m³)' if param_name == 'total_volume' else 'Valeur de wait_time (minutes)'
    ylabel = 'Nombre d\'engins nécessaires' if param_name == 'total_volume' else 'Nombre d\'engins utilisés'
    title = f'Analyse de sensibilité : {param_name}'

    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig)

def create_advanced_gantt_chart(active_engines, project_duration):
    """Create an advanced and interactive Gantt chart"""
    df_gantt = []
    unique_colors = px.colors.qualitative.Plotly

    for idx, (engine, months) in enumerate(active_engines.items()):
        color = unique_colors[idx % len(unique_colors)]

        # Create contiguous intervals
        intervals = []
        start = None
        prev_month = None

        for month in sorted(months):
            if start is None:
                start = month
                prev_month = month
            elif month != prev_month + 1:
                # End of a contiguous interval
                intervals.append({
                    'Engine': engine.split('_')[0],
                    'Start': start - 1,  # Adjust for index
                    'Finish': prev_month,
                    'Color': color
                })
                start = month

            prev_month = month

        # Add the last interval
        if start is not None:
            intervals.append({
                'Engine': engine.split('_')[0],
                'Start': start - 1,  # Adjust for index
                'Finish': prev_month,
                'Color': color
            })

        df_gantt.extend(intervals)

    # Convert to DataFrame
    df = pd.DataFrame(df_gantt)

    # Create Plotly figure with detailed information
    fig = go.Figure(data=[
        go.Bar(
            x=df['Finish'] - df['Start'],  # Bar width
            y=df['Engine'],
            orientation='h',
            base=df['Start'],
            marker_color=df['Color'],
            hovertemplate=
            '<b>Engine</b>: %{y}<br>' +
            '<b>Start Month</b>: %{base}<br>' +
            '<b>End Month</b>: %{base} + %{width}<br>' +
            '<b>Duration</b>: %{width} months<extra></extra>'
        )
    ])

    # Layout customization
    fig.update_layout(
        title='Detailed Gantt Chart of Equipment Deployment',
        xaxis_title='Project Months',
        yaxis_title='Engines',
        height=600,
        barmode='stack',
        xaxis=dict(
            tickvals=list(range(project_duration)),
            ticktext=[f'Month {m+1}' for m in range(project_duration)]
        )
    )

    return fig

def plot_gantt_chart_plotly(model, engine_usage, engine_period_data, project):
    """Affiche un diagramme de Gantt interactif avec Plotly."""
    if model.status != pulp.LpStatusOptimal:
        st.error("Solution non optimale ou problème infaisable.")
        return

    # Collecte des données pour les engins actifs
    active_engines = {}
    for (engine, month), var in engine_usage.items():
        if var.varValue == 1:
            if engine not in active_engines:
                active_engines[engine] = []
            active_engines[engine].append(month)

    # Prepare data for Plotly
    df = []
    for idx, (engine, months) in enumerate(active_engines.items()):
        for month in months:
            df.append({
                'Engine': engine.split('_')[0],
                'Start': month - 1,
                'Finish': month,
                'Duration': 1
            })

    fig = px.timeline(
        pd.DataFrame(df),
        x_start="Start",
        x_end="Finish",
        y="Engine",
        color="Engine",
        title="Equipment Deployment Timeline"
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(
        tickvals=list(range(project["project_duration"])),
        ticktext=[f"Month {m}" for m in range(1, project["project_duration"] + 1)]
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Point d'entrée principal de l'application Streamlit."""
    st.title("Optimisation des Engins de Terrassement")

    # Formulaire pour la modification des paramètres du projet
    st.sidebar.header("Paramètres du Projet")
    project = {
        "total_volume": st.sidebar.number_input("Volume total à déplacer (m³)", value=2000000, min_value=100000),
        "project_duration": st.sidebar.number_input("Durée totale du projet (mois)", value=20, min_value=1),
        "trips_per_minute": st.sidebar.number_input("Tours/minute", value=2, min_value=1),
        "working_days_per_month": st.sidebar.number_input("Jours ouvrés/mois", value=20, min_value=1),
        "wait_time": st.sidebar.number_input("Temps d'attente par défaut (minutes)", value=12, min_value=0),
        "max_simultaneous_engines": st.sidebar.number_input("Nombre max d'engins simultanés", value=12, min_value=1),
        "max_engines_in_gantt": st.sidebar.number_input("Nombre max d'engins dans le Gantt", value=22, min_value=1),
    }

    # Formulaire pour l'ajout/modification des engins
    st.sidebar.header("Engins")
    engines_data = st.sidebar.text_area("Données des engins (format JSON)", value="""
        {"D1.0064": {"capacity": 1.4, "max_usage": 11, "availability_periods": [[1, 11]]}, ...}
    """)
    try:
        engines = eval(engines_data)  # Attention à la sécurité ! En production, utiliser un parser JSON plus robuste.
    except Exception as e:
        st.error(f"Erreur dans le format des données des engins : {e}")
        return

    if st.button("Lancer l'optimisation"):

        # Calcule dynamique du temps de travail comme avant
        for engine, data in engines.items():
            ci = data["capacity"]
            wait_time = project["wait_time"]
            work_time = 7 * 60 * (1 - (wait_time / ((12 / ci) + wait_time)))
            data["work_time_per_day"] = work_time

        engine_period_data = generate_engine_period_combinations(engines)
        model, engine_usage = solve_optimization_problem(project, engine_period_data, engines)

        active_engines = extract_active_engines(engine_usage, engine_period_data)

        generate_professional_excel(project, engines, active_engines, engine_period_data)

        if model.status == pulp.LpStatusOptimal:
            st.success(f"Optimisation réussie ! Nombre d'engins actifs : {pulp.value(model.objective)}")

            plot_gantt_chart_plotly(model, engine_usage, engine_period_data, project)

            # Analyse de sensibilité sur le Volume
            sensitivity_volume_results = []
            variations = [-100000, -50000, 0, 50000, 100000]
            for v in variations:
                mod_proj = project.copy()
                mod_proj["total_volume"] = project["total_volume"] + v
                mod, _ = solve_optimization_problem(mod_proj, engine_period_data, engines)
                sensitivity_volume_results.append((mod_proj["total_volume"], pulp.value(mod.objective)))
            plot_sensitivity_analysis(sensitivity_volume_results, 'total_volume')

            sensitivity_wait_time_results = []
            variations = list(range(-8, 0))  # Variation de -8 à 0 (8 à 16 minutes au total)
            for v in variations:
                modified_project = project.copy()
                modified_project["wait_time"] = 16 + v  # Ajuster pour wait_time entre 8 et 16
                for engine, data in engines.items():
                    ci = data["capacity"]
                    wait_time = modified_project["wait_time"]
                    data["work_time_per_day"] = 7 * 60 * (1 - (wait_time / ((12 / ci) + wait_time)))
                mod, _ = solve_optimization_problem(modified_project, engine_period_data, engines)
                sensitivity_wait_time_results.append((modified_project["wait_time"], pulp.value(mod.objective)))

            plot_sensitivity_analysis(sensitivity_wait_time_results, 'wait_time')

        else:
            st.error("Le problème est infaisable.")

if __name__ == "__main__":
    main()
