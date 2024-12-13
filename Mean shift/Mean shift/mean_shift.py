import numpy as np
tracking_positions = []  # Pour enregistrer les positions (x, y)
tracking_similarities = []  # Pour enregistrer les valeurs de similarité

# import matplotlib.pyplot as plt

class Mean_Shift_Tracker:
    def __init__(self, x_center: int, y_center: int, obj_width: int, obj_height: int):
        # Centre du cadre cible
        self.prev_cx = x_center
        self.prev_cy = y_center
        self.curr_cx = x_center
        self.curr_cy = y_center

        # Coefficient de Bhattacharyya
        self.prev_similarity_BC = 0.0
        self.curr_similarity_BC = 0.0

        # Assurez-vous que la hauteur et la largeur de la cible sont impaires pour faciliter le traitement
        if (obj_width % 2 == 0):
            obj_width += 1
        if (obj_height % 2 == 0):
            obj_height += 1

        self.prev_width = obj_width
        self.prev_height = obj_height
        self.curr_width = obj_width
        self.curr_height = obj_height

        # Définition des paramètres du modèle
        self.bins_per_channel = 16
        self.bin_size = int(np.floor(256 / self.bins_per_channel))
        self.model_dim = self.bins_per_channel ** 3

        # Paramètres du modèle
        self.target_model: np.ndarray = np.zeros(self.model_dim)
        self.prev_model: np.ndarray = np.zeros(self.model_dim)
        self.curr_model: np.ndarray = np.zeros(self.model_dim)

        # Tableau pour attribuer des indices à chaque valeur de pixel dans l'histogramme des couleurs
        self.combined_index: np.ndarray = np.zeros([self.curr_height, self.curr_width])
        self.max_itr = 5

        # Initialiser le noyau du modèle
        self.kernel_mask = self.init_kernel(self.curr_width, self.curr_height)

    def init_kernel(self, w, h):
        """
        Initialiser le noyau de masque du modèle
        """
        half_width = (w - 1) * 0.5
        half_height = (h - 1) * 0.5
        x_limit = int(np.floor((w - 1) * 0.5))
        y_limit = int(np.floor((h - 1) * 0.5))
        x_range: np.ndarray = np.array([range(-x_limit, x_limit + 1)])
        y_range: np.ndarray = np.array([range(-y_limit, y_limit + 1)])
        y_range: np.ndarray = np.transpose(y_range)
        x_matrix: np.ndarray = np.repeat(x_range, y_limit * 2 + 1, axis=0)
        y_matrix: np.ndarray = np.repeat(y_range, x_limit * 2 + 1, axis=1)
        x_square: np.ndarray = x_matrix ** 2
        y_square: np.ndarray = y_matrix ** 2
        x_square = x_square / float(half_width * half_width)
        y_square = y_square / float(half_height * half_height)
        kernel_mask: np.ndarray = np.ones([h, w]) - (y_square + x_square)
        kernel_mask[kernel_mask < 0] = 0

        return kernel_mask

    def update_target_model(self, ref_image):
        """Utiliser l'image entrée pour calculer le modèle initial de la cible de suivi"""
        self.update_object_model(ref_image)
        self.target_model = np.copy(self.curr_model)

    def update_object_model(self, image: np.ndarray):
        """Mettre à jour le modèle de couleur en fonction de l'image entrée"""
        self.curr_model = self.curr_model * 0.0
        self.combined_index = self.combined_index * 0

        # Convertir le type de données de l'image entrée
        image = image.astype(float)
        half_width = int((self.curr_width - 1) * 0.5)
        half_height = int((self.curr_height - 1) * 0.5)

        # Sélectionner la région dans l'image correspondant au cadre de détection
        obj_image = image[self.curr_cy - half_height: self.curr_cy + half_height + 1,
                    self.curr_cx - half_width: self.curr_cx + half_width + 1, :]

        # plt.figure()
        # plt.imshow(obj_image/256)
        # plt.show()

        # Modéliser les couleurs de cette région
        index_matrix = obj_image / self.bin_size  # Réduire la profondeur de couleur à 16 bits
        index_matrix = np.floor(index_matrix).astype(int)
        b_index, g_index, r_index = index_matrix[:, :, 0], index_matrix[:, :, 1], index_matrix[:, :, 2],
        # Générer l'indice de couleur combiné
        combined_index = b_index * np.power(self.bins_per_channel, 2) + \
                         self.bins_per_channel * g_index + \
                         r_index

        self.combined_index = combined_index.astype(int)
        if combined_index.shape != self.kernel_mask.shape:
            kernel_mask = self.init_kernel(combined_index.shape[1], combined_index.shape[0])
        else:
            kernel_mask = self.kernel_mask
        # Mettre à jour le modèle de distribution des couleurs
        for i in range(self.curr_height):
            for j in range(self.curr_width):
                self.curr_model[combined_index[i, j]] += kernel_mask[i, j]
                # Normaliser
        sum_val = np.sum(self.curr_model)
        self.curr_model = self.curr_model / float(sum_val)

    def update_similarity_value(self):
        """
        Calculer et mettre à jour la distance BC entre le cadre précédent et celui-ci
        """
        self.curr_similarity_BC = 0.0
        # Calculer la similarité BC entre deux distributions
        for i in range(self.model_dim):
            if (self.target_model[i] != 0 and self.curr_model[i] != 0):
                self.curr_similarity_BC += np.sqrt(self.target_model[i] * self.curr_model[i])

    def perform_mean_shift(self, image):
        """
        Itération de mean shift
        """
        half_width = (self.curr_width - 1) * 0.5
        half_height = (self.curr_height - 1) * 0.5

        norm_factor = 0.0
        tmp_x = 0.0
        tmp_y = 0.0

        # Initialiser le centre de la boîte de ce tour avec le centre de la dernière boîte
        self.curr_cx = self.prev_cx
        self.curr_cy = self.prev_cy

        # Itération de mise à jour avec l'algorithme mean shift
        for _ in range(self.max_itr):
            # Vérifier la confiance de la cible de ce tour avec la position de la dernière cible
            self.update_object_model(image)
            self.update_similarity_value()
            self.prev_similarity_BC = self.curr_similarity_BC
            feature_ratio = self.target_model / (self.curr_model + 1e-5)

            # Calculer la nouvelle position de la cible
            for i in range(self.curr_height):
                for j in range(self.curr_width):
                    tmp_x += (j - half_width) * feature_ratio[self.combined_index[i, j]]
                    tmp_y += (i - half_height) * feature_ratio[self.combined_index[i, j]]
                    norm_factor += feature_ratio[self.combined_index[i, j]]

            mean_shift_x = tmp_x / norm_factor
            mean_shift_y = tmp_y / norm_factor

            # Mettre à jour la position de la cible avec mean-shift
            self.curr_cx += np.round(mean_shift_x)
            self.curr_cy += np.round(mean_shift_y)
            self.curr_cx = int(self.curr_cx)
            self.curr_cy = int(self.curr_cy)

            # Recalculer le modèle
            self.update_object_model(image)
            # Calculer la similarité
            self.update_similarity_value()
            # Ajuster finement la position du cadre de recherche
            while (self.curr_similarity_BC - self.prev_similarity_BC < -0.01):
                self.curr_cx = int(np.floor((self.curr_cx + self.prev_cx) * 0.5))
                self.curr_cy = int(np.floor((self.curr_cy + self.prev_cy) * 0.5))
                # Vérifier la convergence du modèle
                self.update_object_model(image)
                self.update_similarity_value()
                diff_x = self.prev_cx - self.curr_cx
                diff_y = self.prev_cy - self.curr_cy
                # Calculer la distance euclidienne entre les centres avant et après la mise à jour
                euc_dist = np.power(diff_x, 2) + np.power(diff_y, 2)
                # Vérifier la convergence
                if (euc_dist <= 2):
                    break

            diff_x = self.prev_cx - self.curr_cx
            diff_y = self.prev_cy - self.curr_cy

            # Ajustement fin supplémentaire
            euc_dist = np.power(diff_x, 2) + np.power(diff_y, 2)

            self.prev_cx = self.curr_cx
            self.prev_cy = self.curr_cy

            # Vérification de la convergence
            if (euc_dist <= 2):
                break

        global tracking_positions, tracking_similarities
        tracking_positions.append((self.curr_cx, self.curr_cy))
        tracking_similarities.append(self.curr_similarity_BC)