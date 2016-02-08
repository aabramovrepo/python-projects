/**
  * Class for working with colors that are being used for painting
  *
  * @author Alexey Abramov abramov@physik3.gwdg.de
  */

#ifndef _C_COLOR_H_
#define _C_COLOR_H_

#include <map>

/** @brief Class for storing RGB components */
struct colorRGB{

  int m_R;   //!< R component
  int m_G;   //!< G component
  int m_B;   //!< B component

  /** Constructor by default */
  colorRGB(): m_R(0), m_G(0), m_B(0){}

  /** Constructor with parameters
    * @param compR R component
    * @param compG G component
    * @param compB B component
    */
  colorRGB(const int compR, const int compG, const int compB): m_R(compR), m_G(compG), m_B(compB){}

};

/** @brief Class for generating and storing colors */
class CColor{

private:
  unsigned int m_ColorsNumber;                //!< number of available unique colors 
  std::map<unsigned int,colorRGB> m_Colors;   //!< container with colors: index -> unique RGB color

public:
  /** Constructor by default */
  CColor(){}

  /** Constructor with parameters
   *  @param colors_nmb number of unique colors
   */
  CColor(unsigned int colors_nmb);

  /** Destructor */
  ~CColor();

  /** Get color by index
    * @param index color index
    * @param color structure with RGB components (return)
    */
  bool GetColor(unsigned int index, colorRGB &color);

  /** Get number of available colors
    * @return number of available colors
    */
  int GetColorsAmount() const { return m_Colors.size(); }

};

#endif /* _C_COLOR_H_ */
