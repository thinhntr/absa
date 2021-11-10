# Template file:
- original.txt: chứa data gốc chưa qua preprocessing
- anota_Son.txt: chứa thông tin gán nhãn của Sơn
- anota_Thinh.txt: chứa thông tin gán nhãn của Thịnh  
- cal.ipynb: tính toán độ đồng thuận
### Định dạng chung của file original.txt:

``` 
#id
Nội dung
```
### Định dạng chung của file anota_xxx.txt

```
#id
{nhãn lớp 1, nhãn thuộc tính 1},... {nhãn lớp n, nhãn thuộc tính n}
```
### Định dạng chung của file result.txt
```
#id
{nhãn lớp 1, nhãn thuộc tính 1},... {nhãn lớp n, nhãn thuộc tính n}
```
### List label
Theo định dạng entity-attribute với 3 giá trị có thể nhận là positive/ negative/ neutral 
```
RESTAURANT#GENERAL             
RESTAURANT#PRICES              
RESTAURANT#MISCELLANEOUS       
  
FOOD#QUALITY                   
FOOD#STYLE&OPTIONS             
FOOD#PRICES                    

DRINKS#QUALITY                 
DRINKS#STYLE&OPTIONS           
DRINKS#PRICES

SERVICE#GENERAL                
AMBIENCE#GENERAL
LOCATION#GENERAL
```