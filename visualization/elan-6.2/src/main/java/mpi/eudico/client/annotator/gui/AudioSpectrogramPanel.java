package mpi.eudico.client.annotator.gui;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JPanel;

import mpi.eudico.server.corpora.clom.Transcription;

/**
 * The actual JPanel for rendering the audio spectrogram
 *
 * @author Allan van Hulst
 * @version 1.0
 */
public class AudioSpectrogramPanel extends JPanel {
	
    /**
     * Re-draw the spectrogram.
     * 
     * @param g The graphics context
     */
    public void paintComponent (Graphics g) {
    	super.paintComponent(g);
    	
    	g.drawString("Please specify an interval in the text fields above", 20, 20);
    }
}
