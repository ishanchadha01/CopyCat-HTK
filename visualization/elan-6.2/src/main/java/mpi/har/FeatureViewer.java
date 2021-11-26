package mpi.har;

import mpi.eudico.client.annotator.commands.global.CreateTimeSeriesViewerMA;
import mpi.eudico.client.annotator.viewer.TimeSeriesViewer;
import mpi.eudico.client.annotator.ElanFrame2;
import mpi.eudico.client.annotator.commands.ELANCommandFactory;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.Action;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JCheckBox;

import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;

import java.util.*;

import org.math.plot.*;

public class FeatureViewer {

    public Plot2DPanel plotPanel;
    public HashMap<String, Boolean> timeseriesOptions;
    public String timeseriesData;

    public FeatureViewer() {
        JFrame frame = new JFrame("Feature Viewer Options");
        JPanel panel = new JPanel(new GridLayout());
        JButton fileButton = new JButton("Select File");
        JTextArea textArea = new JTextArea("text");
        
        fileButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (e.getSource() == fileButton) {
                    JFileChooser fc = new JFileChooser();
                    if (fc.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                        File file = fc.getSelectedFile();
                        String fp = file.getPath();
                        try {
                            BufferedReader br = new BufferedReader(new FileReader(fp));
                            String s1 = "";
                            timeseriesData = "";
                            while ((s1=br.readLine()) != null) {
                                timeseriesData += s1+"\n";
                            }
                            br.close();
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                    }
                }
            }
        });

        
        JButton plotDataButton = new JButton("Plot Data");

        plotDataButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (timeseriesData != null && e.getSource() == plotDataButton) {
                    // createTimeSeriesViewer();
                    System.out.println(timeseriesData);
                    ElanFrame2 frame = new ElanFrame2();
                    // Add listener for frame or figure out how to update view?
                    // Add viewer manager with empty dummy transcription
                    // create timeseries viewer for viewer manager
                    // use java awt transforms to help with plotting on timeseries viewer
                    TimeSeriesViewer tsViewer = new TimeSeriesViewer();
                } else if (e.getSource() == plotDataButton) {
                    // Tell user to add file
                }
            }
        });

        
        JButton performAnalysisButton = new JButton("Perform Analysis");
        performAnalysis(this.timeseriesOptions, this.timeseriesData);



        panel.add(fileButton);
        panel.add(plotDataButton);
        panel.add(textArea);
        frame.setSize(new Dimension(600, 400));
        panel.setPreferredSize(new Dimension(600, 400));
        frame.setContentPane(panel);
        frame.pack();
        frame.setVisible(true);
    }

    public void performAnalysis(HashMap<String,Boolean> options, String data) {

    }

    public void createPlotPanel() {
        double[] x = {1.,2.,3.,4.,5.};
        double[] y = {5.,4.,3.,2.,1.};
        this.plotPanel = new Plot2DPanel();
        this.plotPanel.addLinePlot("Test", x, y);
        JFrame frame = new JFrame("Plot Panel");
        frame.setContentPane(plotPanel);
        frame.setVisible(true);
    }

    public class Gaussian { 

        protected double stdDeviation, variance, mean; 
     
        public Gaussian(double stdDeviation, double mean) { 
            this.stdDeviation = stdDeviation; 
            variance = stdDeviation * stdDeviation; 
            this.mean = mean; 
        } 
     
        public double getY(double x) {
            return Math.pow(Math.exp(-(((x - mean) * (x - mean)) / ((2 * variance)))), 1 / (stdDeviation * Math.sqrt(2 * Math.PI))); 
        }
    }
}
