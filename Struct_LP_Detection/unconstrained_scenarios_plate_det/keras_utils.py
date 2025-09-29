import numpy as np
import cv2
import time

from os.path import splitext

from .label import Label
from .utils import nms, im2single
from .projection_utils import getRectPts, find_T_matrix


class DLabel (Label):

	def __init__(self,cl,pts,prob):
		self.pts = pts
		tl = np.amin(pts,1)
		br = np.amax(pts,1)
		Label.__init__(self,cl,tl,br,prob)

#def save_model(model,path,verbose=0):
#	path = splitext(path)[0]
#	model_json = model.to_json()
#	with open('%s.json' % path,'w') as json_file:
#		json_file.write(model_json)
#	model.save_weights('%s.h5' % path)
#	if verbose: print(  'Saved to %s' % path )
#
#def load_model(path,custom_objects={},verbose=0):
#
#	path = splitext(path)[0]
#	with open('%s.json' % path,'r') as json_file:
#		model_json = json_file.read()
#	model = model_from_json(model_json, custom_objects=custom_objects)
#	model.load_weights('%s.h5' % path)
#	if verbose: print ( 'Loaded from %s' % path )
#	return model


# A função agora precisa receber os parâmetros do letterboxing
def reconstruct(Iorig, Yr, threshold, pad_x, pad_y, ratio, out_size, model_size):
    
    # --- O início da função permanece o mesmo ---
    net_stride = 2**4
    side = ((208. + 40.)/2.)/net_stride
    Probs = Yr[...,0]
    Affines = Yr[...,2:]
    
    # Pega as dimensões do canvas a partir do tamanho do modelo
    model_h, model_w = model_size
    MN = np.array([model_w, model_h], dtype=float) / net_stride
    
    xx,yy = np.where(Probs>threshold)
    
    vxx = vyy = 0.5
    base = lambda vx,vy: np.matrix([[-vx,-vy,1.],[vx,-vy,1.],[vx,vy,1.],[-vx,vy,1.]]).T
    labels = []

    for i in range(len(xx)):
        y,x = xx[i],yy[i]
        affine = Affines[y,x]
        prob = Probs[y,x]
        mn = np.array([float(x) + .5, float(y) + .5])
        A = np.reshape(affine,(2,3))
        A[0,0] = max(A[0,0],0.)
        A[1,1] = max(A[1,1],0.)
        
        pts = np.array(A*base(vxx,vyy))
        pts_MN_center_mn = pts*side
        pts_MN = pts_MN_center_mn + mn.reshape((2,1))
        
        # As coordenadas ainda são relativas ao canvas aqui
        pts_prop = pts_MN/MN.reshape((2,1))
        
        labels.append(DLabel(0,pts_prop,prob))
    
    final_labels = nms(labels,.1)
    TLps = []

    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for i,label in enumerate(final_labels):
            
            # --- Início da Modificação Crucial ---
            
            # 1. Converte as coordenadas (0-1) para o tamanho do canvas
            pts_canvas = label.pts * np.array([[model_w],[model_h]])
            
            # 2. Remove o padding
            pts_unpadded = pts_canvas - np.array([[pad_x],[pad_y]])
            
            # 3. Reverte o escalonamento para o tamanho original
            ptsh = pts_unpadded / ratio
            
            # Concatena a linha de '1's para a transformação de perspectiva
            ptsh = np.concatenate((ptsh, np.ones((1,4))))
            
            # --- Fim da Modificação ---
            
            t_ptsh = getRectPts(0,0,out_size[0],out_size[1])
            H = find_T_matrix(ptsh,t_ptsh)
            Ilp = cv2.warpPerspective(Iorig,H,out_size,borderValue=.0)

            TLps.append(Ilp)

    return final_labels,TLps
	

def detect_lp(interpreter, I, out_size, threshold): #<-- Assinatura simplificada
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, model_h, model_w, _ = input_details[0]['shape']
    
    # Normaliza a imagem aqui dentro
    I_normalized = im2single(I)

    # --- Pré-processamento com Letterboxing ---
    img_h, img_w, _ = I_normalized.shape
    ratio = min(model_w / img_w, model_h / img_h)
    new_w, new_h = int(img_w * ratio), int(img_h * ratio)
    Iresized = cv2.resize(I_normalized, (new_w, new_h))

    canvas = np.full((model_h, model_w, 3), 0, dtype=np.float32)
    pad_x = (model_w - new_w) // 2
    pad_y = (model_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = Iresized

    T = np.expand_dims(canvas, axis=0) # O canvas já é float32 e normalizado

    # --- Inferência TFLite ---
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], T)
    interpreter.invoke()
    Yr = interpreter.get_tensor(output_details[0]['index'])
    elapsed = time.time() - start
    Yr = np.squeeze(Yr)

    # --- Pós-processamento ---
    L, TLps = reconstruct(
        Iorig=I,  # <-- MUDANÇA: Passa a imagem original uint8
        Yr=Yr,
        threshold=threshold,
        pad_x=pad_x,
        pad_y=pad_y,
        ratio=ratio,
        out_size=out_size,
        model_size=(model_w, model_h)
    )

    return L, TLps, elapsed
