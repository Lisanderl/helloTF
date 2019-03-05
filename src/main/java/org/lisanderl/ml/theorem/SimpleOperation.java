package org.lisanderl.ml.theorem;

import lombok.extern.java.Log;
import org.tensorflow.Graph;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Subtract;
import org.tensorflow.op.core.Variable;

import java.io.UnsupportedEncodingException;

import static org.lisanderl.ml.theorem.TfHelper.constant;

@Log
public class SimpleOperation {

    public static void main(String... args) throws UnsupportedEncodingException {

        try (Graph g = new Graph()) {
            var scope = new Scope(g);
            var shape = Shape.make(3, 3);
            Variable<Integer> integerVariable = Variable.create(scope, shape, Integer.class);
            Placeholder<Integer> integerPlaceholder = Placeholder.create(scope, Integer.class);

            Constant<Integer> integerConstant = Constant.create(scope, 50);

            Subtract<Integer> subtract = Subtract.create(scope, integerVariable, integerVariable);

            OperationBuilder ob = g.opBuilder("Add", scope.makeOpName("Add"));
            ob.addInput(constant("MyConst1", 2.0F, Float.class, g));
            ob.addInput(constant("MyConst2", 2.0F, Float.class, g));
            ob.build();

            try (var s = new Session(g);

                 Tensor output2 = s.runner().fetch("Add").run().get(0)) {

                System.out.println(output2.floatValue());

            }
        }


        // Execute the "MyConst" operation in a Session.

    }
}

