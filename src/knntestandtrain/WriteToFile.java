/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package knntestandtrain;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import javax.swing.JOptionPane;


/**
 *
 * @author si
 */
public class WriteToFile {
    public static void writeUsingOutputStream(String data){
        OutputStream os = null;
        try{
            os = new FileOutputStream(new File("WYNIK.txt"));
            os.write(data.getBytes(), 0, data.length());
            JOptionPane.showMessageDialog(null,"Wynik zapisny.");
        }catch(IOException e){
            e.printStackTrace();
        }finally{
            try{
                os.close();
            }catch(IOException e){
                e.printStackTrace();
            }
        } 
    }
}
