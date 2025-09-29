import cv2 

def draw_in_image(image, bounding_box, plate, texto):
    """
    Desenha uma bounding box, uma sub-imagem e um texto na imagem principal.

    Args:
        imagem_a (np.ndarray): A imagem principal carregada com OpenCV (em formato BGR).
        bounding_box (list): Uma lista no formato [x, y, w, h], onde (x, y) é o canto
                             superior esquerdo e (w, h) são a largura e a altura.
        imagem_b (np.ndarray): A sub-imagem a ser desenhada no canto inferior esquerdo.
        texto (str): O conteúdo da string a ser escrita acima da bounding box.

    Returns:
        np.ndarray: A imagem 'A' com as anotações desenhadas.
    """
    imagem_anotada = image.copy()

    x, y, w, h = bounding_box
    cor_retangulo = (0, 255, 0)  # Verde em BGR
    espessura_linha = 2
    cv2.rectangle(imagem_anotada, (x, y), (x + w, y + h), cor_retangulo, espessura_linha)

    posicao_texto = (x, y - 10)  # Posição um pouco acima da caixa
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala_fonte = 0.7
    cor_texto = (255, 255, 255)  # Verde em BGR
    espessura_texto = 2
    cor_fundo = (0, 0, 0)



    (largura_texto, altura_texto), baseline = cv2.getTextSize(texto, fonte, escala_fonte, espessura_texto)

    ponto1_retangulo = (posicao_texto[0], posicao_texto[1] - altura_texto - baseline)
    ponto2_retangulo = (posicao_texto[0] + largura_texto, posicao_texto[1] + baseline)

    cv2.rectangle(imagem_anotada, ponto1_retangulo, ponto2_retangulo, cor_fundo, -1) # thickness = -1 preenche
    cv2.putText(imagem_anotada, texto, posicao_texto, fonte, escala_fonte, cor_texto, espessura_texto)

    if plate is not None:
        imagem_anotada[0:plate.shape[0], 0:plate.shape[1]] = plate[:, :]


    return imagem_anotada
