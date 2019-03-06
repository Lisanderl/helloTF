package org.lisanderl.ml.presentation;

import lombok.extern.java.Log;

import java.io.IOException;

@Log
public class SimpleTraining {

    public static void main(String ... args){

        String graphPath = "python\\model\\graph.pb";
        String checkpointPath = "python\\model\\checkpoint";
        Trainer trainer = null;
        try {
            trainer = new Trainer(graphPath, checkpointPath);
        } catch (IOException e) {
            log.info("Probab;y incorrect path " + e);
        }
        trainer.beginTraining("checkpoint1", 100);
    }


}
