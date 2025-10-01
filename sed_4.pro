pro sed_4

; This procedure estimate a rough SED for star+disk and compares it 
; with observations
; Star is simulated as a black body
; The disk is assumed to be optically thick
; 
; Star parameters

teff=4400.     ; effective temperature in K
rstar=1.75     ; stellar radius in solar radii
mstar=1.04     ; stellar mass in solar masses
par=6.61       ; parallax in mas
Av=1.146       ; interstellar absorption in mag
;aa="B"        ; black body spectrum
aa="K"         ; Kurucz model spectrum
mdot=-8.40     ; logarithm of mass accretion rate (in solar Mass/yr)

; Grain properties

alb1=0.5       ; ice albedo
alb2=0.15      ; silicate albedo
tice=170.      ; evaporation temperature for ices (in K)
tsil=1500.     ; evaporation temperature for silicates (in K)
alphasil=-2.64 ; exponent of power law distribution in radius for silicate grains
alphaice=-2.98 ; exponent of power law distribution in radius for ice grains
rmaxgrain=3.  ; maximum grain size (cm)

;rmin=0.1         ; micron
;rmax=10000.      ; micron

rsil=(1.0-alb2)^0.25/tsil^2*rstar*teff^2/2./214.8 ; This radius is equal to the evaporation of silicates
rice=(1.0-alb2)^0.25/tice^2*rstar*teff^2/2./214.8 ; This radius is equal to the ice-line
rice2=(1.0-alb1)^0.25/tice^2*rstar*teff^2/2./214.8 ; This radius is equal to the ice-line for pure ices
print, "Silicate line (au)       =", rsil
print, "Ice line (au)            =", rice
print, "Pure ice line (au)       =", rice2

; Disk parameters (the disk is supposed to be made of two segments)

mdisk=0.001   ; inner disk mass in Msun
mouter=0.078*44.4/231.   ; outer disk mass (within rout2 au) in Msun
rin=rsil       ; inner radius of silicate disk in au
rout1=13.1     ; outer radius of ice disk in au
rin2=28.4      ; inner radius of outer disk segment in au
rout2=44.4     ; outer radius of outer disk segment in au
inc=39.22      ; disk inclination in degree

print, "Inner disk mass (Msun)   =", mdisk
print, "Inner disk lifetime (Myr)=", mdisk/10^mdot/1.0e6

; Reads observational data

dire="C:\Common\sphere\Destinys\WRAY\sed\"
star="sed_data.txt"
readcol, dire+star, wlo,f1,err1,f2,err2, format="(f,f,f,f,f)"
f1=f1+(Av/1.33)*(f2-f1)
errf1=alog10(1.0+err2/f2)

; useful data

nwl=600
wl=findgen(nwl)
wl=0.3+0.01*wl^1.5
wlcm=1e-4*wl
h=6.63e-27
c=2.9978e10
kb=1.38e-16
gc=6.67e-8
msun=1.989e33
dcm=3.09e21/par
cosinc=cos(inc*!pi/180.)
dil=8.0+2.0*alog10(dcm)    ; distance effect; includes the translation from erg/cm^2/s/cm to erg/cm^2/s/A
mouter=mouter/mdisk
mdisk=msun*mdisk

; accretion luminosity

mstarg=msun*mstar
mdotgs=10^mdot*msun/(86400.*365.243)
rstarcm=6.96e10*rstar
lacc=10^(alog10(gc)+alog10(mstarg)+alog10(mdotgs)-alog10(rstarcm))                      

; Star photospheric spectrum

phs=fltarr(nwl)
c1=2.0*h*c^2
c2=h*c/kb

if aa eq "B" then begin
  phs=c1/wlcm^5/(exp(c2/(wlcm*teff))-1.0)
  phs=alog10(!pi*rstarcm^2*phs)
  phs=phs-dil
endif
if aa eq "K" then begin
  dire2=dire+"Kurucz\"
  tmod=3500+250*indgen(27)
  i=0
  while teff gt tmod[i] do i=i+1
  nome1=dire2+"kp00_"+strtrim(tmod[i-1],1)+".fits"
  nome2=dire2+"kp00_"+strtrim(tmod[i],1)+".fits"
  spec=MRDFITS(nome1,1,hdr)
  wave1 = spec.wavelength
  flux1 = spec.g00
  spec=MRDFITS(nome2,1,hdr)
  wave2 = spec.wavelength
  flux2 = spec.g00
  flux=10^(alog10(flux1)+(teff-tmod[i-1])*(alog10(flux2)-alog10(flux1))/(tmod[i]-tmod[i-1]))
  wave=wave1/10000.
  for i=0,nwl-1 do begin
    j=0
    while wl[i] gt wave[j] and j lt 1220 do j=j+1
    phs[i]=flux[j-1]+(wl[i]-wave[j-1])*(flux[j]-flux[j-1])/(wave[j]-wave[j-1])
  endfor
  phs=alog10(rstarcm^2*phs)
  phs=phs-dil+8.
endif

; Accretion spectrum

pha=fltarr(nwl)
tacc=10000.                              ; assumed temperature for the accretion region
pha=c1/wlcm^5/(exp(c2/(wlcm*tacc))-1.0)
lum1=5.6704e-5*tacc^4
surf=lacc/lum1

pha=alog10(pha*surf)
pha=pha-dil

; Disk thermal spectrum 

phd=fltarr(nwl)

; The disk is subdivided into nseg segments. 
; For each segment we consider the emitting area and the flux computed with a Planck formula
; Temperature for each segment is estimated using the equilibrium temperature
; Albedo needed in the formula is the value for ices for distances >1.15*r_ice, else the value for silicates

nseg=470
r1=findgen(nseg)
rstep=0.0005
expo=2.0
r=0.02+rstep*r1^expo
dr=expo*rstep*r1^(expo-1.0)

mtot=0.0
for i=0, nseg-1 do if r[i] gt rsil and r[i] lt rout1 then mtot=mtot+dr[i]
norma=mdisk/mtot/1.495e13
sigmagas=norma/(2.0*!pi*r*1.495e13)

;mtot2=0.0
;for i=0, nseg-1 do if r[i] gt rsil and r[i] lt rout1 then mtot2=mtot2+2.0*!pi*sigmagas[i]*(1.495e13)^2*r[i]*dr[i]

;plot, alog10(r),alog10(sigmagas), yrange=[-3,5]
;oplot, alog10(r),alog10(sigmasil), linestyle=2
;oplot, alog10(r),alog10(sigmaice), linestyle=1
;stop

dg=0.1
fact=10^dg-1.0
ndim=fix(10.*(alog10(rmaxgrain)+5.0))+1
rgrain=dg*findgen(ndim)-5.0     ; cm
rgrain=10^rgrain
dgrain=rgrain*fact
silgrainmass=3.5*(4.0*!pi/3.0)*rgrain^3    ;densities are as in Kataoka et al. 2014, A&A 568, A42
icegrainmass=0.92*(4.0*!pi/3.0)*rgrain^3
mtotsil=silgrainmass*rgrain^alphasil*dgrain
mtotice=icegrainmass*rgrain^alphaice*dgrain
ntotsil=rgrain^alphasil*dgrain
ntotice=rgrain^alphaice*dgrain
ngrainsil=total(ntotsil)
ngrainice=total(ntotice)
ntotsil=ntotsil/ngrainsil
ntotice=ntotice/ngrainice


msil=total(mtotsil)
mice=total(mtotice)
mmedsil=msil/ngrainsil
mmedice=msil/ngrainice


q=fltarr(nwl,ndim)
for j=0, ndim-1 do begin
  for i=0, nwl-1 do begin               ; opacity: formula by Mordasini, 2014, A&A, 572, A118
    xx=0.0001*wl[i]/rgrain[j]
    if xx lt 0.375 then q[i,j]=0.3*xx
    if xx ge 0.375 and xx lt 2.188 then q[i,j]=0.8*xx^2
    if xx ge 2.188 and xx lt 1000. then q[i,j]=2.0+4.0/xx
    if xx ge 1000. then q[i,j]=2.0
    q[i,j]=q[i,j]*!pi*rgrain[j]^2
  endfor
endfor


tseg=fltarr(nseg)
for i=0, nseg-1 do begin
  area=2.0*!pi*r[i]*dr[i]*214.8^2*cosinc
  if r[i] gt rsil then sigmasil=0.0043*sigmagas[i] else sigmasil=0.0 ; Miyake & Nakagawa (1993)
  if r[i] gt rice then sigmaice=0.0094*sigmagas[i] else sigmaice=0.0
  a=fltarr(nwl)
  optthick=fltarr(nwl)
  if r[i] gt rin and r[i] lt rout1 then begin
    for j=0,nwl-1 do begin
      for k=0,ndim-1 do optthick[j]=optthick[j]+q[j,k]*((sigmasil/mmedsil)*ntotsil[k]+(sigmaice/mmedice)*ntotice[k])
      a[j]=area*optthick[j]/(1.0+optthick[j])
    endfor
  endif
  if r[i] gt rin2 and r[i] lt rout2 then begin
    for j=0,nwl-1 do begin
     for k=0,ndim-1 do optthick[j]=optthick[j]+q[j,k]*((sigmasil/mmedsil)*ntotsil[k]+(sigmaice/mmedice)*ntotice[k])
      optthick[j]=mouter*optthick[j]
      a[j]=area*optthick[j]/(1.0+optthick[j])
    endfor
  endif
  if r[i] lt rice then tseg[i]=teff*sqrt(rstar/(2.0*r[i]*214.8))*(1.0-alb2)^0.25 else tseg[i]=teff*sqrt(rstar/(2.0*r[i]*214.8))*(1.0-alb1)^0.25
  for j=0,nwl-1 do phd[j]=phd[j]+a[j]*c1/wlcm[j]^5/(exp(c2/(wlcm[j]*tseg[i]))-1.0)
endfor

; Temperature structure of the disk

i=0
while tseg[i] gt 30.0 do i=i+1
print, "N2 ice line (au)", r[i]

phd=alog10(phd)+2.0*alog10(6.96e10)-dil

; Scattered light by disk (not included yet)

; Total value of the SED

pht=alog10(10^phd+10^phs+10^pha)

; Final plots

plot, alog10(wlo), alog10(f1), psym=4, xrange=[-1,2.5], yrange=[-17,-12], xtitle="log Wavelength (micron)", ytitle="log Flux (erg/cm^2/s/A)"
errplot, alog10(wlo), alog10(f1)+errf1, alog10(f1)-errf1
oplot, alog10(wl), pht, linestyle=0
oplot, alog10(wl), phs, linestyle=1
oplot, alog10(wl), phd, linestyle=2
oplot, alog10(wl), pha, linestyle=3

entry_device = !D.NAME & help, entry_device
set_plot, 'PS'
device, filename=dire+'sed.eps' , /color
plot, alog10(wl), pht, linestyle=0, xrange=[-1,2.5], yrange=[-17,-12], xtitle="log Wavelength (micron)", ytitle="log Flux (erg/cm^2/s/A)"
errplot, alog10(wlo), alog10(f1)+errf1, alog10(f1)-errf1
oplot, alog10(wl), phs, linestyle=1
oplot, alog10(wl), phd, linestyle=2
oplot, alog10(wl), pha, linestyle=3
oplot, alog10(wlo), alog10(f1), psym=4
device, /close_file
set_plot, 'WIN'

stop

end



