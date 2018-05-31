import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage import exposure
import glob, datetime
import clarity.IO as io


 
def manualValidate(cell_centers, img, saveDirectory, nucleusImage=None): 
    '''Pulls up a GUI for manual classification of samples.'''
    ## TODO: make functionality to allow for stopping in the middle 
    ## Currently, this forces users to annotate all of the samples within an image before proceeding.
    # Assume that if nucleusImage is not none, we have a two-channel image 
    
    fig, axes, circles, handles = displayFirstImage(cell_centers[0], img, nucleusImage)
    callback = Index(fig, axes, circles, handles, saveDirectory, cell_centers, img, nucleusImage)
    axprev = plt.axes([0.01, 0.01, 0.18, 0.04])
    axnext = plt.axes([0.20, 0.01, 0.18, 0.04])
    axskip = plt.axes([0.39, 0.01, 0.18, 0.04])
    axsav = plt.axes([0.58, 0.01, 0.18, 0.04])
    axundo = plt.axes([0.77, 0.01, 0.18, 0.04])
    bnext = Button(axnext, 'Positive')
    bnext.on_clicked(callback.positive)
    bprev = Button(axprev, 'Negative')
    bprev.on_clicked(callback.negative)
    bskip = Button(axskip, 'Skip')
    bskip.on_clicked(callback.skip)
    bundo = Button(axundo, 'Undo')
    bundo.on_clicked(callback.undo)
    bsav = Button(axsav, 'Save')
    bsav.on_clicked(callback.save) 
    plt.show()
    return np.load(saveDirectory) # saved state 


    
class Index(object):
    def __init__(self, fig, axes, circles, handles, saveDirectory, cell_centers, img, nucleusImg=None):
        self.ind = 0 
        self.output_labels = []
        self.skipped_inds = [] # list containing all of the skipped indices 
        self.saveDirectory = saveDirectory 
        self.cell_centers = cell_centers 
        self.axes = axes 
        self.handles = handles 
        self.img = img 
        self.nucleusImg = nucleusImg 
        self.circles = circles 
        self.fig = fig 

    def positive(self, event):
        self.ind += 1
        if self.ind == self.cell_centers.shape[0]:
            if len(self.skipped_inds) > 0:
                self.output_labels.append(1)
                self.fig, self.axes, self.circles = displayImage(self.fig, self.axes, self.circles, self.handles, self.skipped_inds[-1], self.cell_centers, self.img, self.output_labels, self.nucleusImg)
            else:
                self.output_labels.append(1)
                np.save(self.saveDirectory, self.output_labels) 
                plt.close('all')
        elif self.ind > self.cell_centers.shape[0]: # This means we are now annotating a skipped image 
            if len(self.skipped_inds) > 0:
                self.output_labels[self.skipped_inds[-1]] = 1
                self.skipped_inds.pop()
                if len(self.skipped_inds) > 0:
                    self.fig, self.axes, self.circles = displayImage(self.fig, self.axes, self.circles, self.handles, self.skipped_inds[-1], self.cell_centers, self.img, self.output_labels, self.nucleusImg)
                else:
                    np.save(self.saveDirectory, self.output_labels)
                    plt.close('all')
            else:
                self.output_labels.insert(self.skipped_inds[-1],1)
                np.save(self.saveDirectory, self.output_labels) 
                plt.close('all')
        else:
            self.output_labels.append(1)
            self.fig, self.axes, self.circles = displayImage(self.fig, self.axes, self.circles, self.handles, self.ind, self.cell_centers, self.img, self.output_labels, self.nucleusImg)

    def negative(self, event):
        self.ind += 1
        if self.ind == self.cell_centers.shape[0]:
            if len(self.skipped_inds) > 0:
                self.output_labels.append(0)
                self.fig, self.axes, self.circles= displayImage(self.fig, self.axes, self.circles, self.handles, self.skipped_inds[-1], self.cell_centers, self.img, self.output_labels, self.nucleusImg)
            else:
                self.output_labels.append(0)
                np.save(self.saveDirectory, self.output_labels) 
                plt.close('all')
        elif self.ind > self.cell_centers.shape[0]: # This means we are now annotating a skipped image 
            if len(self.skipped_inds) > 0:
                self.output_labels[self.skipped_inds[-1]] = 0 
                self.skipped_inds.pop()
                if len(self.skipped_inds) > 0: # if even after popping we still have skipped indices, then we display that 
                    self.fig, self.axes, self.circles= displayImage(self.fig, self.axes, self.circles, self.handles, self.skipped_inds[-1], self.cell_centers, self.img, self.output_labels, self.nucleusImg)
                else:
                    np.save(self.saveDirectory, self.output_labels)
                    plt.close('all') 
            else:
                self.output_labels.insert(self.skipped_inds[-1],0)
                np.save(self.saveDirectory, self.output_labels) 
                plt.close('all')
        else:
            self.output_labels.append(0)
            self.fig, self.axes, self.circles= displayImage(self.fig, self.axes, self.circles, self.handles, self.ind, self.cell_centers, self.img, self.output_labels, self.nucleusImg)

    def skip(self, event):
        self.ind += 1
        if self.ind > self.cell_centers.shape[0]:
            if len(self.skipped_inds) > 0:
                self.fig, self.axes, self.circles = displayImage(self.fig, self.axes, self.circles, self.handles, self.skipped_inds[-1], self.cell_centers, self.img, self.output_labels, self.nucleusImg)
        else:
            self.output_labels.append(-1)
            self.skipped_inds.append(self.ind-1)
            if self.ind == self.cell_centers.shape[0]:
                self.fig, self.axes, self.circles= displayImage(self.fig, self.axes, self.circles, self.handles, self.skipped_inds[-1], self.cell_centers, self.img, self.output_labels, self.nucleusImg)
            else:
                self.fig, self.axes, self.circles = displayImage(self.fig, self.axes, self.circles, self.handles, self.ind, self.cell_centers, self.img, self.output_labels, self.nucleusImg)
                
    def undo(self, event):
        if self.ind == 0:
            print("You're at the beginning of the annotation set.") 
        else:
            self.ind -= 1
            if self.ind > self.cell_centers.shape[0] - 1:
                self.output_labels[self.skipped_inds[-1]] = -1 
            else:
                self.output_labels.pop()
                self.fig, self.axes, self.circles = displayImage(self.fig, self.axes, self.circles, self.handles, self.ind, self.cell_centers, self.img, self.output_labels, self.nucleusImg)
    
    def save(self, event):
        np.save(self.saveDirectory, self.output_labels)
    
    ## TODO: allow for save and quitting; incomplete labeling - currently does not 
    
def displayFirstImage(points, image, nucleusImage):
    # Display the first image to create the figure handles. 
    # points is of size (3,) which is the first cell center to be displayed 
    def _valid(point, axis, bound_size):
        if point < bound_size:
            point = bound_size
        if point+bound_size > image.shape[axis]:
            point = image.shape[axis]-bound_size-1
        return int(point)
    
    print('Positives: {}, Negatives: {}, Total: {}'.format(0,0,0))
    
    axes = []; circles = []; handles = []
    
    if nucleusImage is None: # Only display the cell type channel 
        fig = plt.figure(figsize=(5,8))
    
        bound_size = 16 # zoomed in (32,32,1)
        xx, x = _valid(points[0], 0, bound_size), int(points[0])
        yy, y = _valid(points[1], 1, bound_size), int(points[1])
        zz, z = _valid(points[2], 2, bound_size), int(points[2])
        
        ax1 = fig.add_subplot(321)
        h1 = ax1.imshow((image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),cmap='Greys_r')
        circle1 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax1.add_artist(circle1)
        ax1.set_title("xy-plane (cell type)")
        circles.append(circle1); axes.append(ax1); handles.append(h1) 
        
        ax2 = fig.add_subplot(322)
        h2 = ax2.imshow((image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),cmap='Greys_r')
        circle2 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax2.add_artist(circle2)
        ax2.set_title("xz-plane (cell type)")
        circles.append(circle2); axes.append(ax2); handles.append(h2) 
        
        bound_size = 32 # zoom out by one (64,64)
        xx, x = _valid(points[0], 0, bound_size), int(points[0])
        yy, y = _valid(points[1], 1, bound_size), int(points[1])
        zz, z = _valid(points[2], 2, bound_size), int(points[2])
           
        ax3 = fig.add_subplot(323)
        h3 = ax3.imshow((image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),cmap='Greys_r')
        circle3 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4) 
        ax3.add_artist(circle3)
        circles.append(circle3); axes.append(ax3); handles.append(h3)
        
        ax4 = fig.add_subplot(324)
        h4 = ax4.imshow((image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),cmap='Greys_r')
        circle4 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax4.add_artist(circle4)
        circles.append(circle4); axes.append(ax4); handles.append(h4)
        
        bound_size = 64 # zoom out by one (128,128)
        xx, x = _valid(points[0], 0, bound_size), int(points[0])
        yy, y = _valid(points[1], 1, bound_size), int(points[1])
        zz, z = _valid(points[2], 2, bound_size), int(points[2])
           
        ax5 = fig.add_subplot(325)
        h5 = ax5.imshow((image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),cmap='Greys_r')
        circle5 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4) 
        ax5.add_artist(circle5)
        circles.append(circle5); axes.append(ax5); handles.append(h5)
        
        ax6 = fig.add_subplot(326)
        h6 = ax6.imshow((image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),cmap='Greys_r')
        circle6 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax6.add_artist(circle6)
        circles.append(circle6); axes.append(ax6); handles.append(h6)
        
    else:
        fig = plt.figure(figsize=(10,8))
        
        bound_size = 16
        xx, x = _valid(points[0], 0, bound_size), int(points[0])
        yy, y = _valid(points[1], 1, bound_size), int(points[1])
        zz, z = _valid(points[2], 2, bound_size), int(points[2])
        
        ax1 = fig.add_subplot(341)
        h1 = ax1.imshow((image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),cmap='Greys_r')
        circle1 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax1.add_artist(circle1)
        ax1.set_title("xy-plane (cell type)")
        axes.append(ax1); circles.append(circle1); handles.append(h1)
        
        ax2 = fig.add_subplot(342)
        h2 = ax2.imshow((image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),cmap='Greys_r')
        circle2 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax2.add_artist(circle2)
        ax2.set_title("xz-plane (cell type)")
        axes.append(ax2); circles.append(circle2); handles.append(h2) 
        
        ax3 = fig.add_subplot(343)
        h3 = ax3.imshow((nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),cmap='Greys_r')
        circle3 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax3.add_artist(circle3)
        ax3.set_title("xy-plane (nucleus)")
        axes.append(ax3); circles.append(circle3); handles.append(h3)
        
        ax4 = fig.add_subplot(344)
        h4 = ax4.imshow((nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),cmap='Greys_r')
        circle4 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4) 
        ax4.add_artist(circle4)
        ax4.set_title("xz-plane (nucleus)")
        axes.append(ax4); circles.append(circle4); handles.append(h4)
        
        bound_size = 32 # zoomed out (64,64) 
        xx, x = _valid(points[0], 0, bound_size), int(points[0])
        yy, y = _valid(points[1], 1, bound_size), int(points[1])
        zz, z = _valid(points[2], 2, bound_size), int(points[2])
           
        ax5 = fig.add_subplot(345)
        h5 = ax5.imshow((image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),cmap='Greys_r')
        circle5 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax5.add_artist(circle5)
        axes.append(ax5); circles.append(circle5); handles.append(h5)
        
        ax6 = fig.add_subplot(346)
        h6 = ax6.imshow((image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),cmap='Greys_r')
        circle6 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax6.add_artist(circle6)
        axes.append(ax6); circles.append(circle6); handles.append(h6) 
        
        ax7 = fig.add_subplot(347)
        h7 = ax7.imshow((nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),cmap='Greys_r')
        circle7 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax7.add_artist(circle7)
        axes.append(ax7); circles.append(circle7); handles.append(h7)
        
        ax8 = fig.add_subplot(348)
        h8 = ax8.imshow((nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),cmap='Greys_r')
        circle8 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4) 
        ax8.add_artist(circle8)
        axes.append(ax8); circles.append(circle8); handles.append(h8)
    
        bound_size = 64
        xx, x = _valid(points[0], 0, bound_size), int(points[0])
        yy, y = _valid(points[1], 1, bound_size), int(points[1])
        zz, z = _valid(points[2], 2, bound_size), int(points[2])
           
        ax9 = fig.add_subplot(349)
        h9 = ax9.imshow((image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),cmap='Greys_r')
        circle9 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax9.add_artist(circle9)
        axes.append(ax9); circles.append(circle9); handles.append(h9)
        
        ax10 = fig.add_subplot(3,4,10)
        h10 = ax10.imshow((image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),cmap='Greys_r')
        circle10 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax10.add_artist(circle10)
        axes.append(ax10); circles.append(circle10); handles.append(h10) 
        
        ax11 = fig.add_subplot(3,4,11)
        h11 = ax11.imshow((nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),cmap='Greys_r')
        circle11 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        ax11.add_artist(circle11)
        axes.append(ax11); circles.append(circle11); handles.append(h11)
        
        ax12 = fig.add_subplot(3,4,12)
        h12 = ax12.imshow((nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),cmap='Greys_r')
        circle12 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4) 
        ax12.add_artist(circle12)
        axes.append(ax12); circles.append(circle12); handles.append(h12) 
    return fig, axes, circles, handles
        
        
def displayImage(fig, axes, circles, handles, i, points, image, output_labels, nucleusImage):
    ''' Display an image for the user to manually annotate. '''
    # fig is the figure handle for the plot 
    # i = index of the cell centers that we should annotate 
    # axes = list of the ax handles 
    # points = array containing the cell center coordinates 
    def _valid(point, axis, bound_size):
        if point < bound_size:
            point = bound_size
        if point+bound_size > image.shape[axis]:
            point = image.shape[axis]-bound_size-1
        return int(point)
    
    positive_counts = len(np.argwhere(np.asarray(output_labels)==1))
    negative_counts = len(np.argwhere(np.asarray(output_labels)==0))
    print('Positives: {}, Negatives: {}, Total: {}'.format(positive_counts,negative_counts,positive_counts+negative_counts))
                    
    if nucleusImage is None: # Only display the cell type channel 
        bound_size = 16
        xx, x = _valid(points[i,0], 0, bound_size), int(points[i,0])
        yy, y = _valid(points[i,1], 1, bound_size), int(points[i,1])
        zz, z = _valid(points[i,2], 2, bound_size), int(points[i,2])
        
        handles[0].set_data(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z])
        handles[0].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),vmax=np.amax(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]))
        handles[0].set_cmap('Greys_r')
        circles[0].remove()
        del circles[0]
        circle1 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(0, circle1) 
        axes[0].add_artist(circle1)
        
        handles[1].set_data(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size])
        handles[1].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),vmax=np.amax(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        handles[1].set_cmap('Greys_r')
        circles[1].remove()
        del circles[1]
        circle2 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(1, circle2) 
        axes[1].add_artist(circle2)
        
        bound_size = 32
        xx, x = _valid(points[i,0], 0, bound_size), int(points[i,0])
        yy, y = _valid(points[i,1], 1, bound_size), int(points[i,1])
        zz, z = _valid(points[i,2], 2, bound_size), int(points[i,2])
        
        handles[2].set_data(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z])        
        handles[2].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),vmax=np.amax(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]))
        handles[2].set_cmap('Greys_r')
        circles[2].remove()
        del circles[2]
        circle3 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(2, circle3) 
        axes[2].add_artist(circle3)
        
        handles[3].set_data(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size])
        handles[3].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),vmax=np.amax(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        handles[3].set_cmap('Greys_r')
        circles[3].remove()
        del circles[3]
        circle4 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(3, circle4) 
        axes[3].add_artist(circle4)
        
        bound_size = 64
        xx, x = _valid(points[i,0], 0, bound_size), int(points[i,0])
        yy, y = _valid(points[i,1], 1, bound_size), int(points[i,1])
        zz, z = _valid(points[i,2], 2, bound_size), int(points[i,2])
           
        handles[4].set_data(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z])
        handles[4].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]),vmax=np.amax(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]))
        handles[4].set_cmap('Greys_r')
        circles[4].remove()
        del circles[4]
        circle5 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(4, circle5) 
        axes[4].add_artist(circle5)
        
        handles[5].set_data(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size])
        handles[5].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]),vmax=np.amax(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        handles[5].set_cmap('Greys_r')
        print(np.amax(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        circles[5].remove()
        del circles[5]
        circle6 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(5, circle6) 
        axes[5].add_artist(circle6)
        
    else:
        
        bound_size = 16 # zoomed out 
        xx, x = _valid(points[i,0], 0, bound_size), int(points[i,0])
        yy, y = _valid(points[i,1], 1, bound_size), int(points[i,1])
        zz, z = _valid(points[i,2], 2, bound_size), int(points[i,2])
        
        handles[0].set_data(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z])
        handles[0].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]), vmax=np.amax(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]))
        handles[0].set_cmap('Greys_r')
        circles[0].remove()
        del circles[0]
        circle1 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(0, circle1) 
        axes[0].add_artist(circle1)
        
        handles[1].set_data(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size])
        handles[1].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]), vmax=np.amax(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        handles[1].set_cmap('Greys_r')
        circles[1].remove()
        del circles[1]
        circle2 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(1, circle2) 
        axes[1].add_artist(circle2)
        
        handles[2].set_data(nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z])
        handles[2].set_clim(vmin=np.amin(nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]), vmax=np.amax(nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]))
        handles[2].set_cmap('Greys_r')
        circles[2].remove()
        del circles[2]
        circle3 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(2, circle3) 
        axes[2].add_artist(circle3)
        
        handles[3].set_data(nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size])
        handles[3].set_clim(vmin=np.amin(nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]), vmax=np.amax(nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        handles[3].set_cmap('Greys_r')
        circles[3].remove()
        del circles[3]
        circle4 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(3, circle4) 
        axes[3].add_artist(circle4)
        
        bound_size = 32 # zoomed out 
        xx, x = _valid(points[i,0], 0, bound_size), int(points[i,0])
        yy, y = _valid(points[i,1], 1, bound_size), int(points[i,1])
        zz, z = _valid(points[i,2], 2, bound_size), int(points[i,2])
           
        handles[4].set_data(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z])
        handles[4].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]), vmax=np.amax(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]))
        handles[4].set_cmap('Greys_r')
        circles[4].remove()
        del circles[4]
        circle5 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(4, circle5)
        axes[4].add_artist(circle5)
        
        handles[5].set_data(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size])
        handles[5].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]), vmax=np.amax(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        handles[5].set_cmap('Greys_r')
        circles[5].remove()
        del circles[5]
        circle6 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(5, circle6) 
        axes[5].add_artist(circle6)
        
        handles[6].set_data(nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z])
        handles[6].set_clim(vmin=np.amin(nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]), vmax=np.amax(nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]))
        handles[6].set_cmap('Greys_r')
        circles[6].remove()
        del circles[6]
        circle7 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(6, circle7) 
        axes[6].add_artist(circle7)
        
        handles[7].set_data(nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size])
        handles[7].set_clim(vmin=np.amin(nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]), vmax=np.amax(nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        handles[7].set_cmap('Greys_r')
        circles[7].remove()
        del circles[7]
        circle8 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(7, circle8) 
        axes[7].add_artist(circle8)
    
        bound_size = 64
        xx, x = _valid(points[i,0], 0, bound_size), int(points[i,0])
        yy, y = _valid(points[i,1], 1, bound_size), int(points[i,1])
        zz, z = _valid(points[i,2], 2, bound_size), int(points[i,2])
           
        handles[8].set_data(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z])
        handles[8].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]), vmax=np.amax(image[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]))
        handles[8].set_cmap('Greys_r')
        circles[8].remove()
        del circles[8]
        circle9 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(8, circle9)
        axes[8].add_artist(circle9)
        
        handles[9].set_data(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size])
        handles[9].set_clim(vmin=np.amin(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]), vmax=np.amax(image[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        handles[9].set_cmap('Greys_r')
        circles[9].remove()
        del circles[9]
        circle10 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(9, circle10) 
        axes[9].add_artist(circle10)
        
        handles[10].set_data(nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z])
        handles[10].set_clim(vmin=np.amin(nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]), vmax=np.amax(nucleusImage[xx-bound_size:xx+bound_size,yy-bound_size:yy+bound_size,z]))
        handles[10].set_cmap('Greys_r')
        circles[10].remove()
        del circles[10]
        circle11 = plt.Circle((y-yy+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(10, circle11) 
        axes[10].add_artist(circle11)
        
        handles[11].set_data(nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size])
        handles[11].set_clim(vmin=np.amin(nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]), vmax=np.amax(nucleusImage[xx-bound_size:xx+bound_size,y,zz-bound_size:zz+bound_size]))
        handles[11].set_cmap('Greys_r')
        circles[11].remove()
        del circles[11]
        circle12 = plt.Circle((z-zz+bound_size, x-xx+bound_size), 8, color='magenta', alpha = 0.4)
        circles.insert(11, circle12) 
        axes[11].add_artist(circle12)
        
    fig.canvas.draw()
    return fig, axes, circles
    
