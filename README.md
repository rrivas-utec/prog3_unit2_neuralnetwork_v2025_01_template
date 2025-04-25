# Task #EX1: Templates  
**course:** Programaci�n III  
**unit:** 1  
**cmake project:** prog3_unit2_neuralnetwork_v2025_01
## Indicaciones Espec�ficas
El tiempo l�mite para la evaluaci�n es de 2 horas.

Cada pregunta deber� ser respondida en un archivo fuente (.cpp) y cabecera (.h) correspondiente, en caso de `templates` solo incluir el archivo cabecera:

 - `neural_network.h`  

Deber�s subir estos archivos directamente a www.gradescope.com o se puede crear un .zip que contenga todos ellos y subirlo.

## Question: templates (20 points)

## Objetivo
Dise�ar e implementar un mini-framework en C++ que permita construir redes neuronales gen�ricas y modulares a partir de capas definidas por el usuario, usando:

- **variadic templates** y **template-template parameters**
- **Jerarqu�a polim�rfica gen�rica** (`Layer<T>`)
- **Memoria din�mica segura** (`std::unique_ptr`)
- Operaciones num�ricas b�sicas y **Softmax**

---

## Requisitos

1. **Clase base**  
   Define la interfaz polim�rfica gen�rica:
   ```cpp
   template<typename T>
   class Layer {
   public:
       virtual ~Layer() = default;
       virtual std::vector<T> forward(const std::vector<T>& input) = 0;
   };
   ```

2. **Capas derivadas**  
   Implementa **cuatro** clases que hereden de `Layer<T>`:

   - **`Dense<T>`**
      - Un �nico peso (`weight`) y un �nico sesgo (`bias`).
      - La dimensi�n de entrada = dimensi�n de salida = `input.size()`.
      - `forward(in)` produce:
        ```cpp
        for (size_t i = 0; i < in.size(); ++i)
            out[i] = in[i] * weight + bias;
        ```

   - **`ReLU<T>`**
     ```cpp
     out[i] = std::max(T(0), in[i]);
     ```

   - **`Dropout<T>`**
      - Elimina las neuronas pares de la entrada:
        ```cpp
        // in = {x0,x1,x2,x3,...} -> out = {x1,x3,...}
        ```

   - **`Softmax<T>`**
     ```cpp
     auto m = *std::max_element(in.begin(), in.end());
     for (auto x : in) e.push_back(std::exp(x - m));
     auto sum = std::accumulate(e.begin(), e.end(), T(0));
     for (size_t i = 0; i < e.size(); ++i)
         out[i] = e[i] / sum;
     ```

3. **Determinaci�n de la plantilla `NeuralNetwork`**  
   Aqu� tienes algunos **casos de uso**. A partir de ellos, **deduce**:
   - La firma de la plantilla (`template<...> class NeuralNetwork`).
   - El **orden** y n�mero de **par�metros** que debe recibir el constructor.

   ---
   ### Caso 1: Red simple Dense -> Softmax
   ```cpp
   // Dense<float>(weight=0.5, bias=-0.1) -> Softmax<float>()
   NeuralNetwork<float, Dense, Softmax> model1(
       0.5f,    // Dense<float>::weight
      -0.1f     // Dense<float>::bias
      /* Softmax<float>() sin args */
   );

   std::vector<float> in1(5, 1.0f);
   auto out1 = model1.predict(in1);
   // out1.size()==5  y  sum(out1)==1.0f
   ```

   ---
   ### Caso 2: Dense -> ReLU -> Dense -> Softmax
   ```cpp
   // Dense<double>(0.2, 0.0) -> ReLU<double>() -> Dense<double>(0.2, 0.0) -> Softmax<double>()
   NeuralNetwork<double, Dense, ReLU, Dense, Softmax> model2(
       0.2,   // Dense: weight
       0.0    // Dense: bias
       /* ReLU y Softmax sin args */
   );

   std::vector<double> in2 = {1.0, 2.0, 3.0};
   auto out2 = model2.predict(in2);
   // out2.size()==3, sum approx. 1.0, valores en [0,1]
   ```

   ---
   ### Caso 3: Dense -> Dropout -> ReLU -> Dense -> Softmax
   ```cpp
   // Dense<float>(0.05, -0.02) -> Dropout<float>() -> ReLU<float>() -> Dense<float>(0.05, -0.02) -> Softmax<float>()
   NeuralNetwork<float, Dense, Dropout, ReLU, Dense, Softmax> model3(
       0.05f,    // Dense: weight
      -0.02f     // Dense: bias
       /* Dropout, ReLU, Softmax sin args */
   );

   std::vector<float> in3 = {1,2,3,4,5,6};
   auto out3 = model3.predict(in3);
   // tras Dropout: tama�o 3 -> Softmax: tama�o 3, sum==1.0f
   ```
