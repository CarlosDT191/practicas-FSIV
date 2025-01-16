#pragma once
#include <opencv2/core.hpp>

/**
 * @brief Convierte una imagen en color BGR a HSV.
 * @param img imagen de entrada.
 * @return la imagen de salida.
 */
cv::Mat fsiv_convert_bgr_to_hsv(const cv::Mat &img);

/**
 * @brief Realiza una combinación "hard" entre dos imágenes usando una máscara.
 * La imagen de salida tendrá los contenidos de la imagen primera donde la máscara
 * sea 255 y los de la segunda donde sea 0.
 * @param img1 la primera imagen.
 * @param img2 la segunda imagen.
 * @param mask la máscara 0 (img2) / 255 (img1).
 * @return la imagen resultante de la combinación.
 */
cv::Mat fsiv_combine_images(const cv::Mat &foreground, const cv::Mat &background,
                            const cv::Mat &mask);

/**
 * @brief Crea una máscara para marcar puntos dentro de un rango HSV.
 * @param hsv_img es la imagen de entrada (HSV).
 * @param lower_bound es el límite inferior del rango HSV.
 * @param upper_bound es el límite superior del rango HSV.
 * @return la máscara (0/255).
 */
cv::Mat fsiv_create_mask_from_hsv_range(const cv::Mat &hsv_img,
                                        const cv::Scalar &lower_bound,
                                        const cv::Scalar &upper_bound);

/**
 * @brief Sustituye en fondo de una imagen por otra usando un color clave.
 * @param foreg imagen que representa el primer plano.
 * @param backg imagen que representa el fondo con el que rellenar.
 * @param hue tono del color usado como color clave.
 * @param sensitivity permite ampliar el rango de tono con hue +- sensitivity.
 * @return la imagen con la la composición.
 */
cv::Mat fsiv_apply_chroma_key(const cv::Mat &foreg, const cv::Mat &backg, int hue,
                              int sensitivity);
