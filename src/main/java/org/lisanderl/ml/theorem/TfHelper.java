package org.lisanderl.ml.theorem;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class TfHelper {

   public static <T> Output<T> constant(String name, Object value, Class<T> type, Graph g ) {
        try (Tensor<T> t = Tensor.create(value, type)) {
            return g.opBuilder("Const", name)
                    .setAttr("dtype", DataType.fromClass(type))
                    .setAttr("value", t)
                    .build()
                    .output(0);
        }
    }

    public static <T> Output<T> variable(String name, Object value, Class<T> type, Graph g ) {
        try (Tensor<T> t = Tensor.create(value, type)) {
            return g.opBuilder("Variable", name)
                    .setAttr("dtype", DataType.fromClass(type))
                    .setAttr("value", t)
                    .setAttr("shape", t.shape())
                    .build()
                    .output(0);
        }
    }
}
