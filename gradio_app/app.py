import gradio as gr
import requests

# Configuration de l'URL de l'API
API_URL = "http://localhost:8000"

# Variable globale pour contrôler l'auto-refresh
auto_refresh_active = False

def check_api_health():
    """
    Vérifie la santé de l'API
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"✅ API is healthy\n\nTimestamp: {data['timestamp']}\nScript exists: {data['script_exists']}"
        else:
            return f"⚠️ API returned status code: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure it's running on http://localhost:8000"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def start_training():
    """
    Lance l'entraînement du modèle
    """
    try:
        response = requests.post(f"{API_URL}/train", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return (
                f"✅ Training started successfully!\n\n"
                f"Job ID: {data['job_id']}\n"
                f"Status: {data['status']}\n"
                f"Started at: {data['started_at']}\n\n"
                f"Use the job ID to check status in the 'Check Status' tab.",
                data['job_id']
            )
        else:
            return f"⚠️ Error: {response.status_code} - {response.text}", ""
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure it's running on http://localhost:8000", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def check_training_status(job_id):
    """
    Vérifie le statut d'un entraînement
    """
    if not job_id or job_id.strip() == "":
        return "⚠️ Please enter a valid Job ID"
    
    try:
        response = requests.get(f"{API_URL}/status/{job_id}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            status_emoji = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌"
            }
            
            emoji = status_emoji.get(data['status'], "❓")
            
            result = (
                f"{emoji} Training Status\n\n"
                f"Job ID: {data['job_id']}\n"
                f"Status: {data['status'].upper()}\n"
                f"Started at: {data['started_at']}\n"
            )
            
            if data.get('completed_at'):
                result += f"Completed at: {data['completed_at']}\n"
            
            if data.get('error'):
                result += f"\n❌ Error:\n{data['error']}"
            
            return result
        elif response.status_code == 404:
            return f"⚠️ Job ID not found: {job_id}"
        else:
            return f"⚠️ Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure it's running on http://localhost:8000"
    except Exception as e:
        return f"❌ Error: {str(e)}"
    
def make_prediction(features_json):
    """
    Make a prediction using the API
    """
    try:
        import json
        features = json.loads(features_json)
        
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return f"""✅ Prédiction réussie:
- Classe prédite: {result['prediction']} ({'Défaut' if result['prediction'] == 1 else 'Pas de défaut'})
- Probabilité de défaut: {result['probability']:.4f}
- Seuil utilisé: {result['threshold']}"""
        else:
            return f"❌ Erreur: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Interface Gradio avec onglets
with gr.Blocks(title="XGBoost Training API Interface") as demo:
    gr.Markdown(
        """
        # 🤖 XGBoost Training API Interface
        
        Interface pour gérer l'entraînement du modèle XGBoost via FastAPI
        """
    )
    
    with gr.Tabs():
        with gr.Tab("🔮 Prédiction"):
            gr.Markdown("### Faire une prédiction")
            features_input = gr.Textbox(
                label="Features (JSON format)",
                placeholder='{"AMT_CREDIT": 100000, "AMT_INCOME_TOTAL": 50000, ...}',
                lines=5
            )
            predict_btn = gr.Button("Prédire", variant="primary")
            prediction_output = gr.Textbox(label="Résultat")
            
            predict_btn.click(
                fn=make_prediction,
                inputs=[features_input],
                outputs=[prediction_output]
            )
        with gr.Tab("🏥 Health Check"):
            gr.Markdown("### Vérifier l'état de l'API")
            health_output = gr.Textbox(
                label="API Status",
                lines=5,
                interactive=False
            )
            health_btn = gr.Button("Check API Health", variant="primary")
            health_btn.click(
                fn=check_api_health,
                outputs=health_output
            )
        
        with gr.Tab("🚀 Start Training"):
            gr.Markdown(
                """
                ### Lancer l'entraînement du modèle
                
                Cliquez sur le bouton pour démarrer un nouvel entraînement.
                Vous recevrez un Job ID pour suivre la progression.
                """
            )
            train_output = gr.Textbox(
                label="Training Response",
                lines=8,
                interactive=False
            )
            job_id_output = gr.Textbox(
                label="Job ID (copy this for status check)",
                interactive=False
            )
            train_btn = gr.Button("Start Training", variant="primary", size="lg")
            train_btn.click(
                fn=start_training,
                outputs=[train_output, job_id_output]
            )
        
        with gr.Tab("📊 Check Status"):
            gr.Markdown(
                """
                ### Vérifier le statut d'un entraînement
                
                Entrez le Job ID reçu lors du lancement de l'entraînement.
                Cliquez sur "Check Status" pour vérifier manuellement.
                """
            )
            
            with gr.Row():
                job_id_input = gr.Textbox(
                    label="Job ID",
                    placeholder="train_20240101_120000",
                    scale=3
                )
                status_btn = gr.Button("Check Status", variant="primary", scale=1)
            
            status_output = gr.Textbox(
                label="Status Information",
                lines=10,
                interactive=False
            )
            
            # Manual check
            status_btn.click(
                fn=check_training_status,
                inputs=job_id_input,
                outputs=status_output
            )

                
        with gr.Tab("📖 Documentation"):
            gr.Markdown(
                """
                ## API Endpoints
                
                ### 1. Health Check
                - **Endpoint**: `GET /health`
                - **Description**: Vérifie que l'API fonctionne correctement
                
                ### 2. Start Training
                - **Endpoint**: `POST /train`
                - **Description**: Lance l'entraînement du modèle XGBoost
                - **Returns**: Job ID pour suivre la progression
                
                ### 3. Check Status
                - **Endpoint**: `GET /status/{job_id}`
                - **Description**: Récupère le statut d'un entraînement
                - **Statuses**:
                  - ⏳ **pending**: En attente de démarrage
                  - 🔄 **running**: En cours d'exécution
                  - ✅ **completed**: Terminé avec succès
                  - ❌ **failed**: Échec de l'entraînement
                
                ## Comment utiliser
                
                1. **Vérifier l'API**: Allez dans l'onglet "Health Check" et vérifiez que l'API est accessible
                2. **Lancer l'entraînement**: Dans "Start Training", cliquez sur le bouton pour démarrer
                3. **Copier le Job ID**: Copiez le Job ID retourné
                4. **Suivre la progression**: Dans "Check Status", collez le Job ID et cliquez sur "Check Status"
                5. **Rafraîchir**: Cliquez à nouveau sur "Check Status" pour mettre à jour le statut
                
                ## Prérequis
                
                L'API FastAPI doit être lancée sur `http://localhost:8000`
                
                ```bash
                cd c:\\Formation\\Projet_8\\src
                python api.py
                ```
                
                ## Notes
                
                - Rafraîchissez manuellement le statut en cliquant sur "Check Status"
                - L'entraînement peut prendre plusieurs minutes selon la taille des données
                - Vérifiez les logs de l'API FastAPI pour plus de détails en cas d'erreur
                """
            )
    
    gr.Markdown(
        """
        ---
        💡 **Tip**: Gardez l'API FastAPI en cours d'exécution pour utiliser cette interface
        """
    )

demo.launch(
    theme=gr.themes.Soft(),
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)