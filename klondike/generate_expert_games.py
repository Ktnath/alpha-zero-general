from lonelybot_py import collect_training_data_py
import os

def generate_expert_games(num_games: int, output_file: str):
    """Génère des parties d'expert en utilisant collect_training_data avec information parfaite."""
    print(f"🎮 Génération de {num_games} parties d'expert...")
    print(f"📂 Répertoire courant: {os.getcwd()}")
    
    try:
        # Utiliser la fonction Rust optimisée pour générer les parties
        collect_training_data_py(num_games)
        
        # Renommer le fichier généré si nécessaire
        if os.path.exists("training_data.jsonl") and output_file != "training_data.jsonl":
            import shutil
            shutil.move("training_data.jsonl", output_file)
            
        print(f"✅ Génération des parties d'expert terminée")
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération des parties: {str(e)}")

if __name__ == "__main__":
    # Générer 100 parties d'expert
    generate_expert_games(100, "expert_games.jsonl")