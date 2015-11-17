package parser;

import java.io.*;
import java.util.*;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import edu.stanford.nlp.io.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.*;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class StanfordCoreNlpTreeAdder {

  public static void main(String[] args) throws IOException {
    PrintWriter out;
    int i=1;


    StanfordCoreNLP pipeline = new StanfordCoreNLP();
    Annotation annotation;
    for (i=1;i<=55;i++){
    String PATH="../data/example_split_"+i+".json";
	JSONArray obj = new JSONArray();


    JSONArray jsonArray= ReadJson(PATH);
	Iterator<JSONObject> iterator = jsonArray.iterator();

	while (iterator.hasNext()) {
		JSONObject para=iterator.next();
		long imgid = (Long)  para.get("imgid");
		long paraid = (Long)  para.get("paraid");
		String raw = (String) para.get("raw");
		JSONObject tree_obj = new JSONObject();

    	JSONArray list = new JSONArray();
		try{
			annotation = new Annotation(raw);
	    	pipeline.annotate(annotation);
	    	List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);


	    	tree_obj.put("imgid", imgid);
			tree_obj.put("paraid", paraid);
	    	for(CoreMap sentence : sentences) {

	    		list.add(sentence.get(TreeCoreAnnotations.TreeAnnotation.class).toString());
	        }
		}
		catch (Exception e){

			tree_obj.put("imgid", imgid);
			tree_obj.put("paraid", paraid);
			System.out.println("Exception"+e.toString());

		}
    	tree_obj.put("tree_list", list);
		obj.add(tree_obj);
	}



	FileWriter file = new FileWriter("../data/example_tree_"+i+".json");
	file.write(obj.toJSONString());
	file.flush();
	file.close();
	System.out.println("example "+i+" done");
    }
	////////////////////////////////////////////////////////////////


  }
  public static JSONArray ReadJson (String path){


		JSONParser parser = new JSONParser();

		try {

			Object obj = parser.parse(new FileReader(path));

			JSONArray jsonArray = (JSONArray) obj;
			return jsonArray;



		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		}
		return null;
	}

}



