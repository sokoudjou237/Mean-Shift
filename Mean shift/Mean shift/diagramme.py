import os
from pylint.pyreverse.main import Run
from pydotplus import graph_from_dot_file

def generate_class_diagram(file_paths, output_directory='diagrams', project_name='Tracking_video_meanshift'):
    # Assurez-vous que le répertoire de sortie existe
    os.makedirs(output_directory, exist_ok=True)

    # Chemin du fichier dot temporaire
    dot_file = os.path.join(output_directory, f'{project_name}.dot')

    # Générer le fichier .dot avec pyreverse
    Run(['--output-directory', output_directory, '--project', project_name] + file_paths)

    # Utiliser pydotplus pour convertir le fichier .dot en PNG
    graph = graph_from_dot_file(dot_file)
    png_file = os.path.join(output_directory, f'{project_name}.png')
    graph.write_png(png_file)

    print(f'Diagramme de classes généré : {png_file}')

generate_class_diagram(['camera_traking.py'])
