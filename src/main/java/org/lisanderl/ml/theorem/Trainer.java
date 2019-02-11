package org.lisanderl.ml.theorem;

import lombok.NonNull;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

public class Trainer {

    private final byte[] graphDef;
    private final String checkpointDir;

    public Trainer(@NonNull String graphDefPath,
                   @NonNull String checkpointDir) throws IOException {
        graphDef = Files.readAllBytes(Paths.get(graphDefPath));
        this.checkpointDir = checkpointDir;
    }

    public void beginTraining(@NonNull String checkopintName,
                              @NonNull int examples) {

        try (var graph = new Graph();
             var session = new Session(graph);
             var checkpointPrefix = Tensors.create(Paths.get(checkpointDir, checkopintName).toString())) {

            graph.importGraphDef(graphDef);
            session.runner().addTarget("init").run();

            System.out.println("Start values: ");
            printValues(session.runner(), "powVal");

            var random = new Random();
            var powVal = 2;
            var randomMaxVal = 50;

            for (var i = 0; i < examples; i++) {
                var randomB = 5;
              randomB *= randomB;
                var randomC = 4;
               randomC *= randomC;

                System.out.printf("RandomB: %d, RandomC %d \n", randomB, randomC);
                try (var inputTensorB = Tensors.create((float)randomB);
                     var inputTensorC = Tensors.create((float)randomC);
                     var targetTensor = Tensors.create((float) (Math.pow(randomB, powVal)
                             + Math.pow(randomB, powVal)))) {
                    session.runner()
                            .feed("input_b", inputTensorB)
                            .feed("input_c", inputTensorC)
                            .feed("target", targetTensor)
                            .addTarget("train")
                            .run();
                }
            }
            System.out.println();
            System.out.println("End values: ");
            printValues(session.runner(), "powVal");
        }
    }

    private void printValues(@NonNull Session.Runner runner, String... names) {

        for (var name : names) {
            runner = runner.fetch(name + "/read");
        }
        var valuesList = runner.run();
        valuesList.forEach(l -> {
            System.out.println(l.floatValue());
            l.close();
        });
    }


}
