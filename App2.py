import streamlit as st
import pulp
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np
import xlsxwriter
import uuid
from io import BytesIO

# Importer les fonctions d'optimisation de pl_15_2_2.py
from pl_15_2_26 import (
    define_project_parameters, 
    generate_engine_period_combinations, 
    solve_optimization_problem, 
    extract_active_engines
)

class EquipmentManager:
    def __init__(self):
        """Initialiser avec des engins par défaut"""
        self.engines = {
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
        
    def add_engine(self, code, capacity, max_usage):
        """Ajouter un nouvel engin"""
        if code in self.engines:
            st.warning(f"Un engin avec le code {code} existe déjà.")
            return False
        
        self.engines[code] = {
            "capacity": float(capacity),
            "max_usage": int(max_usage),
        }
        return True
    
    def remove_engine(self, code):
        """Supprimer un engin"""
        if code in self.engines:
            del self.engines[code]
            return True
        return False

def perform_sensitivity_analysis(project, engine_period_data, engines, analysis_type):
    """
    Effectuer une analyse de sensibilité avancée
    
    Args:
    - project (dict): Paramètres du projet
    - engine_period_data (dict): Données des périodes des engins
    - engines (dict): Configuration des engins
    - analysis_type (str): Type d'analyse ('volume' ou 'temps_attente')
    
    Returns:
    - tuple: (résultats de l'analyse, figure Plotly)
    """
    results = []
    
    if analysis_type == 'volume':
        variations = [-200000, -100000, 0, 100000, 200000]
        for v in variations:
            mod_proj = project.copy()
            mod_proj["total_volume"] = project["total_volume"] + v
            mod, _ = solve_optimization_problem(mod_proj, engine_period_data, engines)
            results.append({
                'Variation': v, 
                'Volume': mod_proj["total_volume"], 
                'Nombre d\'engins': pulp.value(mod.objective)
            })
        
        df = pd.DataFrame(results)
        fig = px.line(
            df, 
            x='Volume', 
            y='Nombre d\'engins', 
            title='Impact du Volume Total sur le Nombre d\'Engins',
            labels={
                'Volume': 'Volume total à déplacer (m³)', 
                'Nombre d\'engins': 'Nombre d\'engins requis'
            },
            markers=True
        )
        fig.update_layout(
            xaxis_title='Volume total à déplacer (m³)',
            yaxis_title='Nombre d\'engins requis'
        )
    
    elif analysis_type == 'temps_attente':
        variations = list(range(8, 15, 1))  # De 8 à 16 minutes
        for wait_time in variations:
            mod_proj = project.copy()
            mod_proj["wait_time"] = wait_time
            
            # Recalculer les temps de travail
            mod_engines = engines.copy()
            for engine, data in mod_engines.items():
                ci = data["capacity"]
                work_time = 7 * 60 * (1 - (wait_time / ((12 / ci) + wait_time)))
                data["work_time_per_day"] = work_time
            
            mod, _ = solve_optimization_problem(mod_proj, engine_period_data, mod_engines)
            results.append({
                'Temps d\'attente': wait_time, 
                'Nombre d\'engins': pulp.value(mod.objective)
            })
        
        df = pd.DataFrame(results)
        fig = px.line(
            df, 
            x='Temps d\'attente', 
            y='Nombre d\'engins', 
            title='Impact du Temps d\'Attente sur le Nombre d\'Engins',
            labels={
                'Temps d\'attente': 'Temps d\'attente (minutes)', 
                'Nombre d\'engins': 'Nombre d\'engins requis'
            },
            markers=True
        )
        fig.update_layout(
            xaxis_title='Temps d\'attente (minutes)',
            yaxis_title='Nombre d\'engins requis'
        )
    
    return results, fig

def generate_professional_excel(project, engines, active_engines, engine_period_data):
    """Generate a formatted Excel report"""
    # Create a BytesIO object to store the Excel file in memory
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    
    # Define style formats
    header_format = workbook.add_format({"bold": True, "bg_color": "#4F81BD", "font_color": "white", "align": "center", "border": 1})
    cell_format = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})
    total_format = workbook.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1, "align": "center"})
    
    # Project Parameters Sheet
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
    
    # Engine Characteristics Sheet
    worksheet_engines = workbook.add_worksheet("Caractéristiques des Engins")
    worksheet_engines.set_column("A:D", 20)
    
    headers = ["Code Engin", "Capacité (m³)", "Utilisation Max (mois)", "Temps de travail quotidien (minutes)"]
    for col, header in enumerate(headers):
        worksheet_engines.write(0, col, header, header_format)
    for row, (engine, data) in enumerate(engines.items(), start=1):
        worksheet_engines.write(row, 0, engine, cell_format)
        worksheet_engines.write(row, 1, data["capacity"], cell_format)
        worksheet_engines.write(row, 2, data["max_usage"], cell_format)
        worksheet_engines.write(row, 3, round(data.get("work_time_per_day", 0), 2), cell_format)
    
    # Used Engines Sheet
    worksheet_used_engines = workbook.add_worksheet("Engins Utilisés")
    worksheet_used_engines.set_column("A:D", 30)
    worksheet_used_engines.set_column("E:E", 35)
    headers = ["Engin", "Capacité (m³)", "Quantité Transportée (m³)", "Nombre de Mois Utilisés"]
    for col, header in enumerate(headers):
        worksheet_used_engines.write(0, col, header, header_format)
    
    total_transport = 0
    row = 1
    for engine, months in active_engines.items():
        base_engine = engine.split('_')[0]
        capacity = engine_period_data[engine]["capacity"]
        total_volume = (
            capacity
            * engines[base_engine]["work_time_per_day"]
            * project["trips_per_minute"]
            * project["working_days_per_month"]
            * len(months)
        )
        total_transport += total_volume
        worksheet_used_engines.write(row, 0, base_engine, cell_format)
        worksheet_used_engines.write(row, 1, capacity, cell_format)
        worksheet_used_engines.write(row, 2, round(total_volume), cell_format)
        worksheet_used_engines.write(row, 3, len(months), cell_format)
        row += 1
    
    # Total volume row
    worksheet_used_engines.merge_range(row, 0, row, 2, "Total Transporté (m³)", total_format)
    worksheet_used_engines.write(row, 3, round(total_transport), total_format)
    
    workbook.close()
    output.seek(0)
    return output

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



def main():
    st.set_page_config(
        page_title="Optimisation des Engins de Terrassement", 
        page_icon="🚧", 
        layout="wide"
    )
    
    # Initialiser le gestionnaire d'équipement dans st.session_state
    if 'equipment_manager' not in st.session_state:
        st.session_state.equipment_manager = EquipmentManager()
    
    st.title("🚧 Optimisation des Engins de Terrassement")
    
    # Barre latérale pour la gestion des engins
    with st.sidebar:
        st.header("🚜 Gestion des Engins")
        
        # Formulaire d'ajout d'engin
        with st.expander("Ajouter un Nouvel Engin"):
            new_code = st.text_input("Code de l'engin (ex: D1.0064)")
            new_capacity = st.number_input("Capacité (m³)", min_value=0.1, step=0.1)
            new_max_usage = st.number_input("Utilisation maximale (mois)", min_value=1, max_value=20)
            
            
            if st.button("✅ Ajouter l'engin"):
                if new_code and new_capacity > 0 and new_max_usage > 0:
                    success = st.session_state.equipment_manager.add_engine(
                        new_code, new_capacity, new_max_usage
                    )
                    if success:
                        st.success(f"Engin {new_code} ajouté avec succès!")
        
        # Liste des engins avec possibilité de suppression
    st.subheader("Liste des Engins")
    for code in list(st.session_state.equipment_manager.engines.keys()):
        col1, col2 = st.columns([3,1])
        with col1:
            st.write(code)
        with col2:
            if st.button(f"🗑️ ", key=f"remove_{code}"):
                # Remove only the specific engine that was clicked
                del st.session_state.equipment_manager.engines[code]
                st.rerun()



    # Configuration du projet
    st.sidebar.header("📋 Configuration du Projet")
    
    total_volume = st.sidebar.number_input(
        "Volume total à déplacer (m³)", 
        min_value=10000, 
        max_value=10000000, 
        value=2000000, 
        step=10000
    )
    project_duration = st.sidebar.number_input(
        "Durée du projet (mois)", 
        min_value=1, 
        max_value=36, 
        value=20
    )
    max_simultaneous_engines = st.sidebar.number_input(
        "Nombre maximal d'engins simultanés", 
        min_value=1, 
        max_value=50, 
        value=12
    )
    trips_per_minute = st.sidebar.number_input(
        "Tour par minute", 
        min_value=1, 
        max_value=10, 
        value=2
    )
    working_days_per_month = st.sidebar.number_input(
        "Jours de travail par mois", 
        min_value=1, 
        max_value=31, 
        value=20
    )
    wait_time = st.sidebar.number_input(
        "Temps d'attente (minutes)", 
        min_value=0, 
        max_value=60, 
        value=12
    )
    
    # Onglets pour l'optimisation et l'analyse de sensibilité
    tab_opt, tab_sens = st.tabs(["🚀 Optimisation", "📊 Analyse de Sensibilité"])
    
    with tab_opt:
        # Bouton d'optimisation
        if st.button("🔍 Optimiser le Déploiement des Engins"):
            project = {
                "total_volume": total_volume,
                "project_duration": project_duration,
                "trips_per_minute": trips_per_minute,
                "working_days_per_month": working_days_per_month,
                "wait_time": wait_time,
                "max_simultaneous_engines": max_simultaneous_engines,
                "max_engines_in_gantt": 22
            }
            
            engines = st.session_state.equipment_manager.engines
            
            # Calcul dynamique du temps de travail
            for engine, data in engines.items():
                ci = data["capacity"]
                work_time = 7 * 60 * (1 - (wait_time / ((12 / ci) + wait_time)))
                data["work_time_per_day"] = work_time
            
            try:
                engine_period_data = generate_engine_period_combinations(engines)
                model, engine_usage = solve_optimization_problem(project, engine_period_data, engines)
                
                if model.status == pulp.LpStatusOptimal:
                    st.success(f"Optimisation réussie ! Engins actifs : {pulp.value(model.objective)}")
                    
                    active_engines = extract_active_engines(engine_usage, engine_period_data)
                    
                    # Diagramme de Gantt
                    fig_gantt = create_advanced_gantt_chart(active_engines, project_duration)
                    st.subheader("Chronologie du Déploiement des Engins")
                    st.plotly_chart(fig_gantt, use_container_width=True)
                    
                    # Génération du rapport Excel
                    excel_file = generate_professional_excel(project, engines, active_engines, engine_period_data)
                    
                    # Bouton de téléchargement
                    st.download_button(
                        label="📥 Télécharger le Rapport d'Optimisation",
                        data=excel_file,
                        file_name="Rapport_Optimisation_Engins.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                else:
                    st.error("Le problème d'optimisation est irréalisable.")
            
            except Exception as e:
                st.error(f"L'optimisation a échoué : {e}")
    
    with tab_sens:
        st.header("🔬 Analyse de Sensibilité")
        
        sens_type = st.selectbox(
            "Choisissez le type d'analyse",
            ["Volume total", "Temps d'attente"],
            index=0
        )
        
        if st.button("🧪 Lancer l'Analyse de Sensibilité"):
            project = {
                "total_volume": total_volume,
                "project_duration": project_duration,
                "trips_per_minute": trips_per_minute,
                "working_days_per_month": working_days_per_month,
                "wait_time": wait_time,
                "max_simultaneous_engines": max_simultaneous_engines,
                "max_engines_in_gantt": 22
            }
            
            engines = st.session_state.equipment_manager.engines
            
            # Calcul dynamique du temps de travail
            for engine, data in engines.items():
                ci = data["capacity"]
                work_time = 7 * 60 * (1 - (wait_time / ((12 / ci) + wait_time)))
                data["work_time_per_day"] = work_time
            
            try:
                engine_period_data = generate_engine_period_combinations(engines)
                
                analysis_type = 'volume' if sens_type == "Volume total" else 'temps_attente'
                sens_results, sens_fig = perform_sensitivity_analysis(
                    project, 
                    engine_period_data, 
                    engines, 
                    analysis_type
                )
                
                st.plotly_chart(sens_fig, use_container_width=True)
                
                with st.expander("📝 Détails de l'Analyse de Sensibilité"):
                    st.write(pd.DataFrame(sens_results))
            
            except Exception as e:
                st.error(f"L'analyse de sensibilité a échoué : {e}")

if __name__ == "__main__":
    main()
