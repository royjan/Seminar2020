class GraphUtils:
    @staticmethod
    def create_graph(results: list):
        """
        :param results: a list of learning results from ThreadManager
        """
        import numpy as np
        from matplotlib import pyplot as plt
        import matplotlib
        from Algorithm import Model
        fig = plt.figure(figsize=(32, 12))
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}
        matplotlib.rc('font', **font)
        fig.canvas.set_window_title('Results')
        y_label = [item.score for item in results]
        x_label = np.arange(len(y_label))
        models_labels = [Model.get_model_name(clf) for clf in results]
        plt.bar(x_label, y_label)
        plt.xticks(range(len(models_labels)), models_labels)
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title("Results")
        plt.show()
