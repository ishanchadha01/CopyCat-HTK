package nl.mpi.jmmf;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Panel;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.HierarchyEvent;
import java.awt.event.HierarchyListener;

/**
 * A panel that serves as the host (video window, the {@code HWND} handle) for
 * a native MediaFoundation video display control. The video {@code Topology}
 * will only be initialized after this panel has been added to a frame or
 * window and is displayable.
 * 
 * @author Han Sloetjes
 */
@SuppressWarnings("serial")
public class JMMFPanel extends Panel implements ComponentListener, HierarchyListener {
	private final static System.Logger LOG = System.getLogger("NativeLogger");
	private JMMFPlayer player;
	
	/**
	 * Constructor.
	 */
	public JMMFPanel() {
		super(null);
		this.setBackground(new Color(0, 0, 128));
		addComponentListener(this);
		addHierarchyListener(this);
		super.setIgnoreRepaint(true);
	}

	/**
	 * Constructor.
	 * 
	 * @param player the player instance using this panel for display
	 */
	public JMMFPanel(JMMFPlayer player) {
		super(null);
		this.player = player;
		this.setBackground(new Color(0, 0, 128));
		addComponentListener(this);
		addHierarchyListener(this);
		super.setIgnoreRepaint(true);
	}
	
	/**
	 * Sets the player using this panel for video display.
	 * 
	 * @param player the player
	 */
	public void setPlayer(JMMFPlayer player) {
		this.player = player;
	}
	
	/**
	 * Called when this panel is added to a component hierarchy of a frame or
	 * window.
	 */
	@Override
	public void addNotify() {
		super.addNotify();
		//System.out.println("Panel add notify...");
		if (LOG.isLoggable(System.Logger.Level.DEBUG)) {
			LOG.log(System.Logger.Level.DEBUG, "Video panel added to window");
		}
		if (player != null && this.isDisplayable()) {
			//System.out.println("Panel add notify, displayable...");
			if (LOG.isLoggable(System.Logger.Level.DEBUG)) {
				LOG.log(System.Logger.Level.DEBUG, "Setting the video panel for the player");
			}
			player.setVisualComponent(this);
			player.setVisible(true);
			player.repaintVideo();
		} else {
			if (LOG.isLoggable(System.Logger.Level.DEBUG)) {
				LOG.log(System.Logger.Level.DEBUG, "Video panel is not displayable or player is null");
			}
		}
	}

	@Override
	public void removeNotify() {
		if (LOG.isLoggable(System.Logger.Level.DEBUG)) {
			LOG.log(System.Logger.Level.DEBUG, "Video panel removed from window");
		}
		if (player != null) {
			//player.setVisualComponent(null);
			//player.setVisible(false);
		}
		super.removeNotify();
	}

	@Override
	public void componentHidden(ComponentEvent ce) {
		if (player != null) {
			player.setVisible(false);
		}
	}

	@Override
	public void componentMoved(ComponentEvent ce) {
		componentResized(ce);
//		player.repaintVideo();
	}

	@Override
	public void componentResized(ComponentEvent ce) {
		if (player != null && this.isDisplayable()) {
			player.setVisualComponentSize(getWidth(), getHeight());

			player.repaintVideo();
		}
	}

	@Override
	public void componentShown(ComponentEvent ce) {
		componentResized(ce);
//		player.repaintVideo();
	}

	@Override
	public void hierarchyChanged(HierarchyEvent e) {
		if (e.getChangeFlags() == HierarchyEvent.DISPLAYABILITY_CHANGED && isDisplayable()) {
			//System.out.println("Hierarchy...");
			if (player != null) {
				player.setVisualComponent(this);
				player.setVisible(true);
				player.repaintVideo();
			}
		}
	}

	@Override
	public void repaint() {
//		System.out.println("repaint...");
//		//super.repaint();
//		if (player != null) {
//			player.repaintVideo();
//		}
	}

	@Override
	public void paint(Graphics g) {
//		System.out.println("paint...");
//		//super.paint(g);
//		if (player != null) {
//			player.repaintVideo();
//		}
	}

	@Override
	public void update(Graphics g) {
//		System.out.println("update...");
//		//super.update(g);
//		if (player != null) {
//			player.repaintVideo();
//		}
	}

	@Override
	public void paintComponents(Graphics g) {
//		System.out.println("paintComponents...");
//		if (player != null) {
//			player.repaintVideo();
//		}
	}

	@Override
	public void validate() {// validate is called regularly
//		System.out.println("validate...");
//		if (player != null) {
//			player.repaintVideo();
//		}
	}

	@Override
	public void paintAll(Graphics g) {
//		System.out.println("paintAll...");
//		if (player != null) {
//			player.repaintVideo();
//		}
	}

	@Override
	public void repaint(int x, int y, int width, int height) {
//		System.out.println("repaint(xywh)...");
//		if (player != null) {
//			player.repaintVideo();
//		}
	}

	@Override
	public void repaint(long tm, int x, int y, int width, int height) {
//		System.out.println("repaint(txywh)...");
//		if (player != null) {
//			player.repaintVideo();
//		}
	}

	@Override
	public void repaint(long tm) {
//		System.out.println("repaint(t)...");
//		if (player != null) {
//			player.repaintVideo();
//		}
	}

	@Override
	public boolean imageUpdate(Image img, int infoflags, int x, int y, int w,
			int h) {
		return false;
	}

	@Override
	public void setIgnoreRepaint(boolean ignoreRepaint) {

		super.setIgnoreRepaint(true);
	}

}
