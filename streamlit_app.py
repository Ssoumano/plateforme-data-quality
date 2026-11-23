# app.py
"""
Data Quality Platform - PRO
Application Streamlit pour l'analyse de la qualité des données
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Data Quality Platform - PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import des modules utilitaires
import charts
import styles
import utils_data
import utils_openai
import utils_pdf

st.markdown(styles.POWERBI_CSS, unsafe_allow_html=True)


# ==================== INITIALISATION SESSION STATE ====================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'profil' not in st.session_state:
    st.session_state.profil = None
if 'synthesis' not in st.session_state:
    st.session_state.synthesis = None
if 'tests' not in st.session_state:
    st.session_state.tests = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'file_key' not in st.session_state:
    st.session_state.file_key = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None


# ==================== CONFIGURATION OPENAI ====================
openai_client = None
try:
    if "OPENAI_API_KEY" in st.secrets:
        openai_client = utils_openai.get_openai_client(st.secrets["OPENAI_API_KEY"])
    else:
        openai_client = utils_openai.get_openai_client()
except Exception:
    openai_client = None  # clé absente ou invalide : IA désactivée

try:
    openai_model = st.secrets.get("OPENAI_MODEL", utils_openai.DEFAULT_MODEL)
except Exception:
    openai_model = utils_openai.DEFAULT_MODEL


# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Aller à",
    ["📊 Dashboard", "📞 Contact"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Informations sur le statut
if openai_client:
    st.sidebar.success("✅ IA activée")
    send_samples = st.sidebar.checkbox(
        "Envoyer des exemples de valeurs à OpenAI",
        value=False,
        help="Désactivé : seules des statistiques agrégées (noms de colonnes, types, taux de "
             "manquants) sont transmises. Activé : 2-3 valeurs par colonne sont aussi envoyées "
             "à OpenAI. À éviter pour des données personnelles ou confidentielles.",
    )
    st.sidebar.caption("🔒 Les données du fichier ne quittent l'application que lors d'un appel IA.")
else:
    send_samples = False
    st.sidebar.warning("⚠️ IA désactivée")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 À propos")
st.sidebar.info("""
**Data Quality Platform - PRO**

Analysez la qualité de vos données avec des insights IA et des recommandations actionnables.

Version 2.0
""")


# ==================== FONCTIONS UTILITAIRES ====================

def show_fig(fig):
    """Affiche une figure matplotlib puis la ferme (évite la fuite mémoire)."""
    st.pyplot(fig)
    plt.close(fig)



# ==================== PAGE: DASHBOARD ====================

if page == "📊 Dashboard":
    
    # En-tête
    st.title("📊 Data Quality Platform — PRO")
    st.markdown("""
    <p style='font-size: 1.1rem; color: #666; margin-bottom: 2rem;'>
    Importez un fichier CSV ou Excel pour analyser automatiquement la qualité de vos données, 
    obtenir des insights IA et des recommandations actionnables.
    </p>
    """, unsafe_allow_html=True)
    
    # Zone d'upload avec instructions
    st.markdown("### 📁 Importer vos données")
    
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Glissez-déposez votre fichier ici",
            type=["csv", "xlsx", "xls"],
            help="Formats supportés: CSV, XLSX, XLS • Taille max: 200MB",
            label_visibility="collapsed"
        )
    
    with col_info:
        st.info("""
        **💡 Formats acceptés:**
        - CSV (délimiteur auto-détecté)
        - Excel (.xlsx, .xls)
        
        **📊 Analyses incluses:**
        - Profiling automatique
        - Détection d'anomalies
        - Recommandations IA
        """)
    
    # Si aucun fichier
    if uploaded_file is None and st.session_state.df is None:
        if True:
            st.markdown("---")
            st.markdown("### 🎯 Comment ça marche ?")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class='card' style='text-align: center;'>
                    <h2 style='color: #667eea;'>1️⃣</h2>
                    <h4>Importez</h4>
                    <p>Glissez votre fichier CSV ou Excel</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='card' style='text-align: center;'>
                    <h2 style='color: #667eea;'>2️⃣</h2>
                    <h4>Analysez</h4>
                    <p>Obtenez un profiling complet automatique</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class='card' style='text-align: center;'>
                    <h2 style='color: #667eea;'>3️⃣</h2>
                    <h4>Améliorez</h4>
                    <p>Suivez les recommandations IA</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class='card' style='text-align: center;'>
                    <h2 style='color: #667eea;'>4️⃣</h2>
                    <h4>Exportez</h4>
                    <p>Téléchargez votre rapport</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.stop()
    
    # ==================== CHARGEMENT DES DONNÉES ====================
    
    # Chargement + profiling : uniquement quand le fichier change
    if uploaded_file is not None:
        file_key = (uploaded_file.name, uploaded_file.size)
        if file_key != st.session_state.file_key:
            with st.spinner("⏳ Chargement et analyse du fichier..."):
                try:
                    new_df = utils_data.load_dataframe(uploaded_file)
                    new_profil = utils_data.profile_data_quality(new_df)
                except Exception as e:
                    st.error(f"❌ Impossible de lire le fichier : {e}")
                    st.stop()
            st.session_state.df = new_df
            st.session_state.profil = new_profil
            st.session_state.file_key = file_key
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_uploaded = True
            st.session_state.synthesis = None
            st.session_state.tests = None
            st.session_state.cleaning_script = None

    profil = st.session_state.profil
    st.success(
        f"✅ Fichier chargé : **{st.session_state.file_name}** "
        f"({profil['rows']:,} lignes × {profil['cols']} colonnes)"
    )

    df = st.session_state.df
    profil = st.session_state.profil
    
    st.markdown("---")
    
    # ==================== KPIs PRINCIPAUX ====================
    
    st.markdown("### 📈 Indicateurs Clés de Performance")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📊 Lignes", f"{profil['rows']:,}")
    
    with col2:
        st.metric("📋 Colonnes", profil['cols'])
    
    with col3:
        score = profil['global_score']
        delta_color = "normal" if score >= 70 else "inverse"
        st.metric(
            "🎯 Score Global", 
            f"{score}%",
            delta=None,
            delta_color=delta_color
        )
    
    with col4:
        st.metric(
            "❌ Valeurs Manquantes",
            f"{profil['total_missing']:,}",
            delta=f"{profil['missing_rate']:.1f}%",
            delta_color="inverse"
        )
    
    with col5:
        st.metric(
            "🔄 Doublons",
            profil['duplicate_rows'],
            delta=f"{profil['duplicate_rate']:.1f}%",
            delta_color="inverse"
        )
    
    with col6:
        st.metric(
            "⚡ Outliers",
            profil.get('total_outliers', 0)
        )
    
    # Barre de progression du score
    score_color = "#4CAF50" if score >= 80 else "#FFC107" if score >= 60 else "#FF9800" if score >= 40 else "#F44336"
    
    st.markdown(f"""
    <div style='margin-top: 20px;'>
        <div style='background: #e0e0e0; border-radius: 10px; height: 30px; overflow: hidden;'>
            <div style='background: {score_color}; width: {score}%; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; transition: width 0.5s ease;'>
                Score de Qualité: {score}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== ONGLETS D'ANALYSE ====================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Aperçu", 
        "📈 Visualisations", 
        "🔍 Profiling Détaillé",
        "🤖 Synthèse IA",
        "🧪 Tests Suggérés",
        "📄 Export"
    ])
    
    # ====== TAB 1: APERÇU ======
    with tab1:
        st.markdown("### 📋 Aperçu du Dataset")
        
        # Options d'affichage
        col_opt1, col_opt2 = st.columns([1, 3])
        with col_opt1:
            max_rows = min(500, len(df))
            if max_rows > 10:
                n_rows = st.slider("Nombre de lignes", 10, max_rows, min(50, max_rows))
            else:
                n_rows = max_rows
        
        st.dataframe(df.head(n_rows), width="stretch", height=400)
        
        # Informations rapides
        st.markdown("### 📊 Informations Rapides")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("**🗂️ Types de Colonnes:**")
            type_dist = profil.get('type_distribution', {})
            for type_name, count in type_dist.items():
                st.write(f"- {type_name}: {count}")
        
        with col_info2:
            st.markdown("**⚠️ Problèmes Détectés:**")
            if profil['empty_columns']:
                st.write(f"- ❌ {len(profil['empty_columns'])} colonne(s) vide(s)")
            if profil['constant_columns']:
                st.write(f"- ⚠️ {len(profil['constant_columns'])} colonne(s) constante(s)")
            if profil['duplicate_rows'] > 0:
                st.write(f"- 🔄 {profil['duplicate_rows']} ligne(s) dupliquée(s)")
            if not (profil['empty_columns'] or profil['constant_columns'] or profil['duplicate_rows'] > 0):
                st.success("✅ Aucun problème majeur détecté!")

        quick_wins = utils_data.generate_quick_wins(profil)
        if quick_wins:
            st.markdown("### ⚡ Quick Wins")
            for qw in quick_wins:
                with st.expander(f"[{qw['priority']}] {qw['action']}"):
                    st.write(f"**Impact :** {qw['impact']}")
                    st.write(f"**Colonnes :** {qw['columns']}")
                    st.code(qw['code'], language="python")
    
    # ====== TAB 2: VISUALISATIONS ======
    with tab2:
        st.markdown("### 📊 Visualisations Avancées")
        
        # Graphiques côte à côte
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("#### 🔴 Valeurs Manquantes")
            fig_missing = charts.create_missing_chart(profil)
            if fig_missing:
                show_fig(fig_missing)
            else:
                st.success("✅ Aucune valeur manquante significative!")
        
        with col_viz2:
            st.markdown("#### 🎨 Distribution des Types")
            fig_types = charts.create_types_chart(profil['column_types'])
            show_fig(fig_types)
        
        # Heatmap des outliers
        st.markdown("#### ⚡ Heatmap des Outliers (IQR)")
        fig_outliers = charts.create_outliers_heatmap(profil)
        
        if fig_outliers:
            show_fig(fig_outliers)
        else:
            st.info("ℹ️ Aucun outlier détecté ou pas de colonnes numériques.")
        
        # Matrice de corrélation
        if not profil['correlations'].empty:
            st.markdown("#### 🔗 Matrice de Corrélation")
            
            show_fig(charts.create_correlation_chart(profil['correlations']))
    
    # ====== TAB 3: PROFILING DÉTAILLÉ ======
    with tab3:
        st.markdown("### 🔍 Profiling Détaillé par Colonne")
        
        col_select = st.selectbox(
            "📋 Sélectionnez une colonne à analyser",
            list(df.columns),
            help="Choisissez une colonne pour voir ses statistiques détaillées"
        )
        
        if col_select:
            st.markdown(f"#### Analyse de : **{col_select}**")
            
            # Informations de base
            col_info_a, col_info_b, col_info_c, col_info_d = st.columns(4)
            
            with col_info_a:
                st.metric("Type Détecté", profil['column_types'].get(col_select, "unknown"))
            
            with col_info_b:
                missing_pct = profil['missing_pct'][col_select]
                st.metric("Valeurs Manquantes", f"{missing_pct:.1f}%")
            
            with col_info_c:
                st.metric("Valeurs Uniques", df[col_select].nunique(dropna=True))
            
            with col_info_d:
                st.metric("Valeurs Totales", len(df[col_select]))
            
            st.markdown("---")
            
            # Visualisations selon le type
            if pd.api.types.is_numeric_dtype(df[col_select]):
                col_num1, col_num2 = st.columns(2)
                
                fig_dist, fig_box = charts.create_numeric_charts(df[col_select], col_select)
                with col_num1:
                    st.markdown("**📊 Distribution**")
                    show_fig(fig_dist)
                
                with col_num2:
                    st.markdown("**📦 Boxplot**")
                    show_fig(fig_box)
                
                # Statistiques descriptives
                st.markdown("**📈 Statistiques Descriptives**")
                st.dataframe(df[col_select].describe().to_frame().T, width="stretch")
                
            else:
                # Pour colonnes non-numériques
                st.markdown("**📊 Distribution des Valeurs**")
                fig_cat, value_counts = charts.create_category_chart(df[col_select], col_select)
                show_fig(fig_cat)
                
                st.dataframe(value_counts.to_frame("Fréquence"), width="stretch")
    
    # ====== TAB 4: SYNTHÈSE IA ======
    with tab4:
        st.markdown("### 🤖 Synthèse & Recommandations IA")
        
        if not openai_client:
            st.warning("""
            ⚠️ **Fonctionnalité IA désactivée**
            
            Pour activer les synthèses et recommandations IA, configurez votre clé API OpenAI dans les secrets Streamlit.
            """)
            
            # Afficher quand même une synthèse basique
            st.markdown("#### 📊 Résumé Rapide (Sans IA)")
            st.info(f"""
            - **Dataset:** {profil['rows']:,} lignes × {profil['cols']} colonnes
            - **Score global:** {profil['global_score']}%
            - **Problèmes:** {len(profil['empty_columns']) + len(profil['constant_columns'])} colonnes problématiques, {profil['duplicate_rows']} doublons
            """)
            
        else:
            if st.button("🚀 Générer la Synthèse IA", type="primary", width="stretch"):
                with st.spinner("🤖 Génération de la synthèse en cours... (cela peut prendre 10-20 secondes)"):
                    try:
                        synthesis = utils_openai.generate_synthesis(
                            openai_client, df, profil,
                            include_samples=send_samples, model=openai_model
                        )
                        st.session_state.synthesis = synthesis
                        st.success("✅ Synthèse générée avec succès!")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération: {str(e)}")
            
            # Afficher la synthèse si elle existe
            if st.session_state.synthesis:
                st.markdown("---")
                st.markdown(st.session_state.synthesis)
                
                # Bouton de régénération
                if st.button("🔄 Régénérer", help="Générer une nouvelle synthèse"):
                    st.session_state.synthesis = None
                    st.rerun()

            st.markdown("---")
            st.markdown("#### 🧹 Script de nettoyage")
            st.caption("Généré à partir des problèmes détectés (noms de colonnes, sans valeurs).")
            if st.button("🧹 Générer un script de nettoyage", width="stretch"):
                with st.spinner("🤖 Génération du script..."):
                    try:
                        st.session_state.cleaning_script = utils_openai.generate_cleaning_script(
                            openai_client, df, profil, model=openai_model
                        )
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération: {str(e)}")
            if st.session_state.get("cleaning_script"):
                script = st.session_state.cleaning_script
                st.code(script, language="python")
                st.download_button("⬇️ Télécharger le script", script.encode("utf-8"),
                                   file_name="nettoyage.py", mime="text/x-python")
    
    # ====== TAB 5: TESTS SUGGÉRÉS ======
    with tab5:
        st.markdown("### 🧪 Tests de Qualité Suggérés")
        
        if not openai_client:
            st.warning("⚠️ Fonctionnalité IA désactivée. Configurez OpenAI pour générer des tests.")
        else:
            if st.button("🧪 Générer les Tests IA", type="primary", width="stretch"):
                with st.spinner("🤖 Génération des tests en cours..."):
                    try:
                        tests = utils_openai.generate_tests(
                            openai_client, df,
                            include_samples=send_samples, model=openai_model
                        )
                        st.session_state.tests = tests
                        st.success("✅ Tests générés avec succès!")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération: {str(e)}")
            
            if st.session_state.tests:
                st.markdown("---")
                st.markdown(st.session_state.tests)
    
    # ====== TAB 6: EXPORT ======
    with tab6:
        st.markdown("### 📄 Export du Rapport")
        
        st.info("""
        💡 **Options d'export disponibles:**
        - Rapport PDF
        - Rapport HTML complet
        - Données nettoyées (CSV : colonnes vides/constantes et doublons retirés)
        """)
        
        synthesis_text = st.session_state.synthesis or (
            "Synthèse non générée. Utilisez l'onglet 'Synthèse IA' pour générer une analyse complète."
        )
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        col_export1, col_export2, col_export3 = st.columns(3)

        with col_export1:
            try:
                st.download_button(
                    "📄 Rapport PDF",
                    data=utils_pdf.generate_pdf_bytes(synthesis_text, profil),
                    file_name=f"rapport_data_quality_{stamp}.pdf",
                    mime="application/pdf",
                    type="primary",
                    width="stretch",
                )
            except Exception as e:
                st.error(f"❌ Erreur PDF : {e}")

        with col_export2:
            st.download_button(
                "🌐 Rapport HTML",
                data=utils_pdf.generate_pdf_html(synthesis_text, profil, df).encode("utf-8"),
                file_name=f"rapport_data_quality_{stamp}.html",
                mime="text/html",
                width="stretch",
            )

        with col_export3:
            cleaned = utils_data.clean_dataframe(df)
            st.download_button(
                "💾 Données nettoyées (CSV)",
                data=cleaned.to_csv(index=False).encode('utf-8'),
                file_name=f"data_cleaned_{stamp}.csv",
                mime="text/csv",
                width="stretch",
            )
            st.caption(
                f"{len(df) - len(cleaned):,} ligne(s) et "
                f"{df.shape[1] - cleaned.shape[1]} colonne(s) retirées."
            )


# ==================== PAGE: CONTACT ====================

elif page == "📞 Contact":
    
    st.title("📞 Contact")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; 
                border-radius: 16px; 
                color: white; 
                margin-bottom: 30px;'>
        <h2 style='margin-bottom: 20px;'>👋 Parlons de vos données!</h2>
        <p style='font-size: 1.1rem; opacity: 0.95;'>
        Besoin d'aide avec votre projet data? D'une consultation sur la qualité de vos données?
        Ou simplement envie de discuter des dernières tendances en Data Quality?
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_contact1, col_contact2 = st.columns([1, 1])
    
    with col_contact1:
        st.markdown("""
        ### 📧 Informations de Contact
        
        **👤 Nom:** SOUMANO Seydou
        
        **📧 Email:** [soumanoseydou@icloud.com](mailto:soumanoseydou@icloud.com)
        
        **📱 Téléphone:** [+33 6 64 67 88 87](tel:+33664678887)
        
        **💼 LinkedIn:** [seydou-soumano](https://linkedin.com/in/seydou-soumano)
        
        **💻 GitHub:** [Ssoumano](https://github.com/Ssoumano)
        """)
    
    with col_contact2:
        st.markdown("### 💡 Domaines d'Expertise")
        st.markdown("""
        - 🔍 Data Quality & Governance
        - 📊 Data Engineering
        - 🤖 Machine Learning & IA
        - 📈 Business Intelligence
        - 🔧 Automatisation & DevOps
        """)
    
    st.markdown("---")
    
    st.markdown("### ✉️ Envoyer un message")
    st.markdown(
        "Écrivez-moi directement : "
        "[soumanoseydou@icloud.com](mailto:soumanoseydou@icloud.com?subject=Data%20Quality%20Platform)"
    )


# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Data Quality Platform - PRO</strong> | Version 2.0</p>
    <p>Développé avec ❤️ par Seydou SOUMANO | © 2026</p>
</div>
""", unsafe_allow_html=True)
