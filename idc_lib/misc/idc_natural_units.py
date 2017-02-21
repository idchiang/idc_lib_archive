from __future__ import absolute_import, division, print_function, \
                       unicode_literals
range = xrange
from astropy.constants import c, k_B, hbar
import astropy.units as u

us = [['Length', 1. / hbar / c, 1. / u.eV, '1/eV',
       [u.um, u.cm, u.m, u.km, u.pc, u.kpc, u.Mpc],
       ['um', 'cm', 'm', 'km', 'pc', 'kpc', 'Mpc']], 
      ['Mass', c**2, u.eV, 'eV', [u.kg, u.g], ['kg', 'g']],
      ['Time', 1. / hbar, 1. / u.eV, '1/eV', [u.s], ['s']], 
      ['Temperature', k_B, u.eV, 'eV', [u.K], ['K']]]

def to_natural_units():
    ## type
    print('Input type: ', end='')
    for i in range(len(us)):
        print(str(i) + ': ' + str(us[i][0]) + '. ', end='')
    print('Others: Pass.')
    t = int(input())
    ## value
    print('Input value:')
    val = float(input())
    ## unit
    print('Input unit: ', end='')
    for i in range(len(us[t][4])):
        print(str(i) + ': ' + str(us[t][5][i]) + '. ', end='')
    print('Others: Error.')
    ui = int(input())
    ans = round((val * us[t][4][ui] * us[t][1]).to(us[t][2]).value, 3)
    print(val, str(us[t][5][ui]), '=', ans, str(us[t][3]))
    
def to_normal_units():
    ## type
    print('Input type: ', end='')
    for i in range(len(us)):
        print(str(i) + ': ' + str(us[i][0]) + '. ', end='')
    print('Others: Pass.')
    t = int(input())
    ## value
    print('Input value (eV or 1/eV):')
    val = float(input())
    ## unit
    print('Input unit: ', end='')
    for i in range(len(us[t][4])):
        print(str(i) + ': ' + str(us[t][5][i]) + '. ', end='')
    print('Others: Error.')
    ui = int(input())
    ans = (val * us[t][2] / us[t][1]).to(us[t][4][ui]).value
    print(val, str(us[t][3]), '=', ans, str(us[t][5][ui]))