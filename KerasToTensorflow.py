import tensorflow as tf

# Set learning phase to 0 (for inference)
tf.keras.backend.set_learning_phase(0)

# Assuming you've loaded the DenseNet169 model as 'loaded_model' in a previous code snippet
# If not, load the model using: loaded_model = tf.keras.models.load_model('path_to_your_model')

model_path = '/Users/darshvaidya/projects/covid_detection/model/modelCovid19___1.h5'

# Load the pre-trained model
loaded_model = tf.keras.models.load_model(model_path)

# Save as a SavedModel
export_dir = '/Users/darshvaidya/projects/covid_detection/model/'
loaded_model.save(export_dir, save_format='tf')

# Load the SavedModel
loaded = tf.saved_model.load(export_dir)

# Create a concrete function with the expected signature
concrete_func = loaded.signatures["serving_default"]

# Freeze saved model
output_node_names = [output.name.split(":")[0] for output in loaded_model.outputs]
frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
    tf.compat.v1.Session(),
    concrete_func.graph.as_graph_def(),
    output_node_names
)

# Write frozen graph to a file
tf.io.write_graph(graph_or_graph_def=frozen_graph, logdir=".", name="frozen_graph_densenet.pb", as_text=False)

# Compress frozen graph into tar.gz file
import tarfile
with tarfile.open("frozen_graph_densenet.tar.gz", "w:gz") as tar:
    tar.add("frozen_graph_densenet.pb")