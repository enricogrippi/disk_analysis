# # Script for DEPROJECTION of circumstellar disks  [ [ Gabriele Columba] ]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage.transform as sktr
from scipy import ndimage
from skimage.transform import warp_polar
from astropy.io import fits
from PIL import Image
from cv2 import GaussianBlur
import vip_hci as vip
import argparse
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/gabriele/ExoDisks/gab_products/bdhammel-ellipse_fitting/')

hd34700_parx = 2.85
zimp_scale = 3.516

def show_steps( orig, rot1, depitch, final, scale_lims=None):
	'''
	Plot the intermediate results of the deprojection procedure.
	'''
	fig, axs = plt.subplots(2,2, figsize=(10, 8), constrained_layout=True)
	axs[0,0].matshow( orig, origin='lower', cmap='inferno', vmin=scale_lims[0], vmax=scale_lims[1])
	axs[0,1].matshow( rot1, origin='lower', cmap='inferno', vmin=scale_lims[0], vmax=scale_lims[1])
	axs[1,0].matshow( depitch, origin='lower', cmap='inferno', vmin=scale_lims[0], vmax=scale_lims[1])
	axs[1,1].matshow( final, origin='lower', cmap='inferno', vmin=scale_lims[0], vmax=scale_lims[1])
	axs[1,1].scatter( (final.shape[1]-1)/2, (final.shape[0]-1)/2, c='r', marker='+')
	axs[0,0].set( title= 'original image' )
	axs[0,1].set( title= 'rotated image' )
	axs[1,0].set( title= 'flaring-corrected' )
	axs[1,1].set( title= 'final image (re-oriented)')
	# r = 45
	# theta = np.linspace(0, 2*3.14, 80)
	# x = r* np.cos(theta)
	# y = r* np.sin(theta)
	# ax4.plot(x + x0_s, y + y0_s, color='white', ls= '--', alpha=0.3)
	plt.show()

def au_to_pix_zimp( au, scale=zimp_scale, parallax=hd34700_parx):
    return au * parallax / scale

def pix_to_au_zimp( pix, scale=zimp_scale, parallax=hd34700_parx):
    return pix * scale / parallax

def au_format_zimp( value, pos):
    """The two arguments are the value and tick position"""
    y_au = pix_to_au_zimp( value)
    return f'{y_au :3.0f}'

def au_deg_ticks_zimp( ax_obj , shift):
    '''Format the ticks to put au on the Y and degrees on the X, for horizontal polar images.'''
    def deg_format_zimp( value, pos):
        x_deg = value + shift
        return f'{x_deg :.0f}'

    ax_obj.yaxis.set_major_locator( mpl.ticker.MultipleLocator( au_to_pix_zimp( 50) ))     # y axis
    ax_obj.yaxis.set_minor_locator( mpl.ticker.MultipleLocator( au_to_pix_zimp( 25) ))
    ax_obj.yaxis.set_major_formatter( au_format_zimp )   
    ax_obj.xaxis.set_major_locator( mpl.ticker.MultipleLocator( 90 ))     # x axis
    ax_obj.xaxis.set_minor_locator( mpl.ticker.MultipleLocator( 45 ))
    ax_obj.xaxis.set_major_formatter( deg_format_zimp )
    return


def display_polar_image( data, polar_shift = -90 , ylims=[0,250], extend=180, colmap='Greys_r'):
	'''Produce a nice figure for the polar domain image.'''
	if extend != 0:
		data = np.concatenate( (data, data[:, 0:extend]), axis=1 )		# extend the image repeating more angles

	ptitle = 'Polar domain image HD34700'
	fig, ax = plt.subplots( figsize=(8,6) )
	#scale_lims = [ 0, 1]
	ax.imshow( data, cmap=colmap, origin='lower', aspect='auto')#, vmin=scale_lims[0], vmax=scale_lims[1])
	ax.set(  xlabel= '$\phi$ (°)', ylabel='$ r $ (au)', frame_on=False)

	ax.grid(True, alpha=0.4, linestyle=':')
	ymin, ymax = ylims[0], ylims[1]    # frame row limits to focus the view
	ax.set_ylim( ymin, ymax)
	au_deg_ticks_zimp( ax, polar_shift )
	#ax.yaxis.set_major_locator( mpl.ticker.MultipleLocator( au_to_pix_zimp( 25) ))
	plt.gcf().canvas.manager.set_window_title( ptitle )
	plt.show()

def deflare_disc( data, flaring, inc, far_side='down'):
	'''
	Correct the data to consider the flaring angle of the disc, in approx of constant slope.
	'''
	if far_side == 'up': sign = +1
	elif far_side == 'down': sign = -1
	dim_y, dim_x = data.shape[0], data.shape[1]
	xx, yy = np.meshgrid( np.arange(dim_x), np.arange(dim_y))
	xx__ = xx - (dim_x - 1)/2
	yy__ = yy - (dim_y - 1)/2
	A = np.sqrt( xx__**2 + yy__**2/ np.cos(inc)**2 )        # for each pixel compute the a of the ellipse it belongs to
	Wcents = A * np.tan(flaring) * np.sin(inc) * sign                    # for each a compute the ellipse center shift due to the flaring
	deflare = ndimage.map_coordinates( data, [ yy + Wcents , xx], order=3, cval=np.nan )     # flaring-corrected image
	# nans_len = len( deflare[:,0][ np.isnan(deflare[:,0]) ] ) + 1
	deflare_cut = deflare #[ nans_len :-nans_len, :]                 # cut away the nan values that bug the final rotation
	return deflare_cut

def recenter_array( data):
	if input( 'recenter image ? [y/n]:  ') == 'y':
		([ Xc, Yc]) = ([ float(x) for x in input( 'New center [x_c, y_c]: ').split() ])   # maybe allow float center in future
		# marg = int( input( 'pixel margin from center: ') )
		# raw_data = re_center( raw_data, center=([ Xc, Yc]), margins= [marg, marg] )
		shifts = ( np.array( data.shape) - 1 ) / 2 - ([ Yc, Xc])
		data = vip.preproc.recentering.frame_shift( data, shifts[0], shifts[1], imlib='opencv', interpolation='bicubic')
	return data

# def OLDcrop_n_center( img_array, C_coord=None):
# 	frame = vip.Frame( img_array)
# 	if C_coord == None:
# 		C_coord = [ frame.get_center()[1], frame.get_center()[0] ]
# 	x_size = 2 * min( C_coord[0], img_array.shape[1] - 1 - C_coord[0])
# 	y_size = 2 * min( C_coord[1], img_array.shape[0] - 1 - C_coord[1])
# 	cropsize = int( min( x_size, y_size) + 1 )
# 	if cropsize % 2 == 0: 
# 		cropsize = cropsize - 1
# 	frame.crop( cropsize, xy=C_coord, force=False )
# 	return frame.data

def crop_n_center( img_array, C_coord=None):
	'''centre an odd square to C_coord = [xc, yc]'''
	frame = img_array.copy()
	geom_C = np.roll( (np.array( frame.shape) - 1) / 2, 1)
	if (C_coord == geom_C).all():
		return frame	# good as it is with its geometrical centre
	elif C_coord == None:
		C_coord = tuple(geom_C)
	x_size = 2 * min( C_coord[0], frame.shape[1] - 1 - C_coord[0])
	y_size = 2 * min( C_coord[1], frame.shape[0] - 1 - C_coord[1])
	cropsize = int( min( x_size, y_size) + 1 )
	if cropsize % 2 == 0: 
		cropsize = cropsize - 1
		cropped = vip.preproc.cosmetics.frame_crop( frame, cropsize, cenxy=C_coord, force=True )
		cropped = vip.preproc.recentering.frame_shift( cropped, -0.5, -0.5, imlib='opencv' )
	else: cropped = vip.preproc.cosmetics.frame_crop( frame, cropsize, cenxy=C_coord, force=True )
	return cropped



def smoother( data, sigma_pix):

	kernel_size = (0, 0)	# so cv2 computes it automatically
	data_sm = GaussianBlur( data, ksize=kernel_size, sigmaX= sigma_pix, sigmaY= sigma_pix)
	return data_sm

def r2_correction( image, C_coord=None, polar=True):
	'''
	Multiply by r^2 the values to correct for flux decay.
	xc, yc coordinates of the centre of image in 0-based index.
	'''
	if C_coord == None:
		yc, xc = (np.array( image.shape ) - 1 ) *0.5
	else: 
		[xc, yc] = C_coord
	
	if polar:
		x_arr = np.zeros( image.shape[1] )
		yc = 0
	else:
		x_arr = np.arange( image.shape[1] ) - xc

	xx, yy = np.meshgrid( x_arr, np.arange(image.shape[0]) - yc )
	r2_mask = xx**2 + yy**2

	return image * r2_mask


def calc_f( inc, dl, ds):
	return np.rad2deg( np.arctan( (dl - ds) / (dl + ds) / np.tan( np.deg2rad(inc)) ) )


def calc_ird_centre( zimp_centre, zimp_scale=3.516, ird_scale=12.255, z_frame_centre=np.array([511,511]), i_frame_centre=np.array([511,511])):
    '''
    compute centre coordinates of irdis from zimpol centre, xc, yc.
    '''
    deltas_z = zimp_centre - z_frame_centre
    deltas_i = deltas_z * zimp_scale / ird_scale
    return list( i_frame_centre + deltas_i )

def arch_spiral_mat( a, b, windings):
	theta = np.linspace(0, windings * 2 * np.pi, 1000 * windings)
	r = a + b * theta
	arcspi = np.zeros( (200,200) )
	x = r * np.cos( theta)
	y = r * np.sin( theta)
	arcspi[ np.rint(x).astype(int) + 100, np.rint(y).astype(int) + 100] = 100
	return arcspi

def log_spiral_mat( a=1, k=0.5, windings=2, dim_arr=600):
	theta = np.linspace(0, windings * 2 * np.pi, 1000 * windings)
	r = a * np.exp(k * theta)
	logspi = np.zeros( (dim_arr, dim_arr) )
	x = np.rint( r * np.cos( theta) ).astype(int)
	y = np.rint( r * np.sin( theta) ).astype(int)
	hd = int(dim_arr/2)
	x = x[ abs(x) < hd ]
	y = y[ abs(y) < hd ]
	size = min( len(x), len(y) )
	x = x[ : size-1]
	y = y[ : size-1]
	logspi[ x + hd, y + hd ] = 100
	return logspi

def savefits( data, name, fname):
	if input( f'save {name} to fits file? [y/n]: ') == 'y':
		ID = input( 'extra IDs for your file? : ' )
		fits.writeto( fname.replace( '.fits', f'__{name}_{ID}.fits'), data)
		return print('fits file saved !')



def main( fname, PA_majaxis, inclination, centre, polar_shift, flaring_ang=0., sigma_smooth=0, badpix_thresh=None, r2correction=False, figs=True, rdim=250, phidim=360):
	'''
	Perform deprojection to the face-on view and polar domain transformation.
	'''
	try:		# if fname is a path
		hdul = fits.open( fname )
		hdul.info()
		raw_data = hdul[0].data[:].byteswap().newbyteorder()	# raw frame to process
	except:		#if fname is an array
		raw_data = fname.copy()
	if badpix_thresh != None:
		raw_data = vip.preproc.badpixremoval.cube_fix_badpix_clump( raw_data, bpm_mask=(raw_data >= badpix_thresh), sig=5, min_thr= 4.)   # ci mette una VITA senza bpm
	raw_data = crop_n_center( raw_data, centre )		# centred frame
	inc = np.deg2rad( inclination )							# angles in radians
	flang = np.deg2rad( flaring_ang )

	## rotate image to align major axis horizontally
	rot_ang = 90 - PA_majaxis
	a_rotated = sktr.rotate( raw_data, -rot_ang, order=3, preserve_range=True)          # rotated image

	## reallign ellipses centres with pitch angle
	if flang != 0. :
		deflared = deflare_disc( a_rotated, flaring=flang, inc=inc, far_side='down')
	else: deflared = a_rotated

	## Simulate the 'face-on' view by stretching the image
	stretch_y = round( deflared.shape[0]/np.cos(inc) )
	new_size = ( deflared.shape[1], stretch_y )            # x-axis size is the number of columns (i.e. shape[1])
	stretched_mat = np.array( Image.fromarray(deflared).resize( new_size, Image.BICUBIC) )      # stretched image  (cubic spline interp)

	## Rotate back the image to set the Nord direction vertical again
	rot_back = np.rad2deg( np.arctan( np.cos(inc) * np.tan( np.deg2rad( rot_ang)) ) )      
	back_rotated = sktr.rotate( stretched_mat, rot_back, order=3, preserve_range=True )                # final image  (alligned to original)
	
	deprojected = crop_n_center( back_rotated )		# recenter to square
	if r2correction == True:
		deprojected = r2_correction( deprojected, polar=False)
	if sigma_smooth != 0. :			# perform gaussian smoothing
		deprojected = smoother( deprojected, sigma_smooth)

	polardom = warp_polar( np.array( Image.fromarray(deprojected).rotate( polar_shift +90, resample=Image.BICUBIC)), radius=rdim, output_shape=(phidim, rdim) ).T  # T to have it horizontal
	if figs:
		# scale_lims = [ 0, 1] 	# deprojected.max()/4 ]
		#show_steps( raw_data, a_rotated, deflared, deprojected, scale_lims )
		plt.imshow( deprojected, cmap='inferno', origin='lower')#, vmin=scale_lims[0], vmax=scale_lims[1])
		plt.scatter( (deprojected.shape[1]-1)/2, (deprojected.shape[0]-1)/2, c='r', marker='+')
		plt.show()
		plt.imshow( polardom, cmap='inferno', origin='lower')#, vmin=scale_lims[0], vmax=scale_lims[1])
		plt.show()
	return deprojected, polardom



if __name__ == '__main__':

	parser = argparse.ArgumentParser()		# parsing file and parameters for deprojection
	parser.add_argument('filename', type=str, help='name of the fits file to deproject (default: None)')
	parser.add_argument('PA_a', type=float, default=90., help='Position Angle [deg] of the disk semimajor axis (default: 90°)')
	parser.add_argument('i', type=float, default=38, help='disk LoS inclination [deg] (default: 38°)')
	parser.add_argument('-C', type=float, nargs=2, default=[511,511], help='frame centre coordinates [xc, yc] (default: 511)')
	parser.add_argument('-flarang', type=float, default=0., help='disk flaring angle (default: 0°)')
	parser.add_argument('-polshift', type=float, default=-90., help='polar domain shift angle (default: -90°)')
	parser.add_argument('-smooth', type=float, default=0, help='smoothing sigma (default: 0)')
	parser.add_argument('--r2corr', action='store_true',  help='apply r^2 correction (default: False)')
	parser.add_argument('--save', action='store_true', help='save figures prompt (default: False)')
	args = vars( parser.parse_args() )

	filename = args['filename']
	# pixscale = args['pixscale']
	PA_a = args['PA_a']
	i = args['i']
	Centre = args['C']
	flare_ang = args['flarang']
	pol_shift = args['polshift']
	sig_smooth = args['smooth']
	r2_corr = args['r2corr']
	save_mode = args['save']

	deprojected, polardom = main( filename, PA_a, i, Centre, pol_shift, flare_ang, sigma_smooth=sig_smooth, badpix_thresh=None, r2correction=r2_corr)

	if save_mode:
		savefits( deprojected, 'deproj', filename)
		savefits( polardom, 'polar', filename)
	print('Deprojection completed')


