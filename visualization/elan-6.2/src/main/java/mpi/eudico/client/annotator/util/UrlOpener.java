package mpi.eudico.client.annotator.util;

import java.awt.Desktop;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

/**
 * Utility class to open a URL and log exceptions.
 * A {@code mailto:} URL is directed to the default mail application,
 * all other URL's are directed to the browser.  
 */
public class UrlOpener {
	/**
	 * Opens the URL either via {@link Desktop#mail(URI)} or via 
	 * {@link Desktop#browse(URI)}.
	 * 
	 * @param url the address to open
	 * @throws Exception any exception that can occur
	 */
	public static void openUrl(String url) throws Exception {
		//System.out.println("openUrl: " + url);
		if (url == null) {
            return;
        }
 		
        URI uri;
		try {
			uri = new URI(url);
			if (url.startsWith("mailto:")) {
				Desktop.getDesktop().mail(uri);
			} else {
				Desktop.getDesktop().browse(uri);
			}
		} catch (URISyntaxException use) {
			ClientLogger.LOG.warning("Error opening webpage: " + use.getMessage());
			throw(use);
			//use.printStackTrace();
		} catch (IOException ioe) {
			ClientLogger.LOG.warning("Error opening webpage: " + ioe.getMessage());
			throw(ioe);
			//ioe.printStackTrace();
		}
	}
}
