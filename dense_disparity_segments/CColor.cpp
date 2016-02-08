/**
  * Implementation of class for working with colors for painting
  *
  * @author Alexey Abramov <abramov@physik3.gwdg.de>
  */

#include "CColor.h"

#include <stdlib.h>
#include <iostream>

/*--------------------------------------------------------------------------------------------------------------------------------------*/
CColor::CColor(unsigned int colors_nmb): m_ColorsNumber(colors_nmb) {

  std::cout << "m_ColorsNumber = " << m_ColorsNumber << std::endl;

  unsigned int ind = 0;

  colorRGB color_black(0,0,0);
  colorRGB color_white(255,255,255);

  colorRGB color_red(255,0,0);
  colorRGB color_green(0,255,0);
  colorRGB color_blue(0,0,255);
  colorRGB color_yellow(255,255,0);
  colorRGB color_cyan(0,255,255);
  colorRGB color_magenta(255,0,255);
  colorRGB color_orange(249,121,7);
  colorRGB color_violet(109,21,241);
  colorRGB color_dark_red(155,4,4);
  colorRGB color_dark_blue(12,23,131);
  colorRGB color_dark_yellow(181,175,0);
  colorRGB color_dark_green(15,106,5);
  colorRGB color_dark_orange(157,108,2);
  colorRGB color_dark_cyan(19,125,135);
  colorRGB color_dark_magenta(121,5,95);

  //m_Colors[ind++] = color_white;  //white color is necessary for boundary detection results to achieve the best contrast
  m_Colors[ind++] = color_black;
  m_Colors[ind++] = color_red;
  m_Colors[ind++] = color_green;
  m_Colors[ind++] = color_blue;
  m_Colors[ind++] = color_yellow;
  m_Colors[ind++] = color_cyan;
  m_Colors[ind++] = color_magenta;
  m_Colors[ind++] = color_orange;
  m_Colors[ind++] = color_violet;
  m_Colors[ind++] = color_dark_red;
  m_Colors[ind++] = color_dark_blue;
  m_Colors[ind++] = color_dark_yellow;
  m_Colors[ind++] = color_dark_green;
  m_Colors[ind++] = color_dark_orange;
  m_Colors[ind++] = color_dark_cyan;
  m_Colors[ind++] = color_dark_magenta;

  // generate the rest checking for uniqueness
  while( ind < m_ColorsNumber ){

    int compR = rand()%256;
    int compG = rand()%256;
    int compB = rand()%256;

    bool flag = false;

    for(unsigned int ii = 0; ii < ind; ++ii){

      if( m_Colors[ind].m_R == compR && m_Colors[ind].m_G == compG && m_Colors[ind].m_B == compB )
        flag = true;

    }

    if( !flag && compR != 0 && compG != 0 && compB != 0 ){

      colorRGB clr(compR,compG,compB);
      m_Colors[ind] = clr;
      ind++;
    }

  }

  std::cout << "m_Colors.size() = " << m_Colors.size() << std::endl;

}
/*--------------------------------------------------------------------------------------------------------------------------------------*/
bool CColor::GetColor(unsigned int index, colorRGB &color){

  if( index > (m_Colors.size() - 1) ){

    std::cout << "Color index is out of range ! index = " << index << std::endl;
    return false;//-->
  }

  color =  m_Colors[index];
  
  return true;

}
/*--------------------------------------------------------------------------------------------------------------------------------------*/
CColor::~CColor(){

  m_Colors.clear();

}
/*--------------------------------------------------------------------------------------------------------------------------------------*/
