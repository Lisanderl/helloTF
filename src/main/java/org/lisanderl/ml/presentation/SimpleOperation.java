package org.lisanderl.ml.presentation;

import lombok.extern.java.Log;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Add;

import java.io.UnsupportedEncodingException;

import static org.lisanderl.ml.presentation.TfHelper.constant;

@Log
public class SimpleOperation {

    public static void main(String... args) throws UnsupportedEncodingException {

        try (Graph g = new Graph()) {
            var scope = new Scope(g);
            Add.create(scope,
                    constant("MyConst12", 2.0F, g),
                    constant("MyConst22", 2.0F, g));

            try (var s = new Session(g);
                 Tensor output2 = s.runner().fetch("Add").run().get(0)) {
                System.out.println(output2.floatValue());
            }
        }
    }
}

