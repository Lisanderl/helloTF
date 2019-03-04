package org.lisanderl.ml.theorem;

import lombok.extern.java.Log;
import org.tensorflow.*;

import java.io.IOException;

@Log
public class SimpleOperation {

    public static void main(String ... args) {

        try(Graph g = new Graph()){

          Output<Float> o1 = constant("t1", 22.2F, Float.class, g);
          Output<Float> o2 = constant("t2", 22.2F, Float.class, g);
        }
    }


   static <T> Output<T> constant(String name, Object value, Class<T> type, Graph g ) {
        try (Tensor<T> t = Tensor.<T>create(value, type)) {
            return g.opBuilder("Const", name)
                    .setAttr("dtype", DataType.fromClass(type))
                    .setAttr("value", t)
                    .build()
                    .<T>output(0);
        }
    }


}
