
from multiprocessing import Value
import re
from time import monotonic
from typing import List


class Mobile_operator:
    
    def __init__(self):
        self.name = "Undefined"
        self.call_cost = 0
        self.gb_limit = 0
        self.sms_limit = 0
        self.min_limit = 0
        self.num_obj = 0
        self.lines = list()
        self.dict = {}
        print("Меню: \n# Add\t\t[Название тарифа] [Стоимость тарифа] [Лимит ГБ] [Лимит СМС] [Скидка] -Добавление объекта\n# Delete\t[Название тарифа]\t-Удаление обьекта")
        print("# Search\t[Название тарифа]\t-Поиск объекта\n# List_obj\t\t\t\t-Список объектов\n# Filter\t[Название атрибута] [Знак сравнения] [Значение]\t-Фильтр по атрибуту\n# Average_stat\t[Название атрибута]\t-Средний показатель по атрибуту объекта\n# Close\t\t-Закрыть программу")
        lst = list()
        with open("3.0lab\mobile.txt", "r", encoding='utf-8') as file:
            for line in file:
                if "\n" in line:
                    line = line[:-2]
                lst.append(line)
        for i in range(1,len(lst)):
            line = str(lst[i])
            line = re.split("\s", line)
            line = list(filter(None,line))
            self.dict[line[0]] = {
                        "call_cost": line[1],
                        "gb_limit": line[2],
                        "sms_limit": line[3],
                        "min_limit": line[4]
                    }
        
        # for key, val in self.dict.items():

        # print(self.dict["Базовый"]["call_cost"])
    # /////////////////////////////////////////////////     Добавляет в список элементы словаря
    def dict_lines(self,dict):
        res = ""
        self.lines.clear()
        for key in dict:
            res += f"{key}\t"
            for value in dict.values():
                if dict[key] == value:
                    for i in value:
                        res += f"{value[i]}\t\t\t"
            self.lines.append(res)
            res = ""
        return self.lines
    # /////////////////////////////////////////////////
    # /////////////////////////////////////////////////     Добавляет объект в словарь[dict] и текстовый документ[mobile.txt]
    def add(self): 
        
        self.dict[self.name] = {
                        "call_cost": self.call_cost,
                        "gb_limit": self.gb_limit,
                        "sms_limit": self.sms_limit,
                        "min_limit": self.min_limit
            }
        lines = self.dict_lines(self.dict)
        with open("3.0lab\mobile.txt", "a", encoding='utf-8') as file:
            lines.reverse()
            file.write(f"\n{lines[0]}")
            lines.reverse()
    # /////////////////////////////////////////////////
    # /////////////////////////////////////////////////     Удаляет объект из словаря[dict] и перезаписывает текстовый документ[mobile.txt]
    def delete(self):
        del self.dict[self.name]
        lines = self.dict_lines(self.dict)
        with open("3.0lab\mobile.txt", "w", encoding='utf-8') as file:
            file.write("Тариф       Стоимость   Лимит_ГБ    Лимит_СМС   Лимит_мин")
            for line in lines:
                file.write(f"\n{line}")
    # /////////////////////////////////////////////////
    # /////////////////////////////////////////////////     Поиск тарифа[name] в словаре[dict] и вывод в консоль
    def search(self):
        lines = self.dict_lines(self.dict)
        for line in lines:
            if self.name in line:
                print("Тариф       Стоимость   Лимит_ГБ    Лимит_СМС   Лимит_мин")
                print(line)
    # /////////////////////////////////////////////////
    # /////////////////////////////////////////////////     Вывод информации о тарифах
    def list_obj(self):
        lines = self.dict_lines(self.dict)
        print("Тариф       Стоимость   Лимит_ГБ    Лимит_СМС   Лимит_мин")
        for line in lines:
            print(line)
    # /////////////////////////////////////////////////
    # /////////////////////////////////////////////////     Фильтрация по атрибутам(Стоимость тарифа, лимит Гб и тд.) по словарю и вывод в консоль
    def filter(self,attr, sign, value):
        l_dict = {}
        value = float(value)
        for i in self.dict:
            for key, val in self.dict[i].items():
                val = float(val)
                if key == attr and sign == ">":
                    if val > value:
                        l_dict[i] = self.dict[i]
                if key == attr and sign == "<":
                    if val < value:
                        l_dict[i] = self.dict[i]
                if key == attr and sign == ">=":
                    if val >= value:
                        l_dict[i] = self.dict[i]
                if key == attr and sign == "<=":
                    if val <= value:
                        l_dict[i] = self.dict[i]
        lines = self.dict_lines(l_dict)
        for line in lines:
            print(line)
    # /////////////////////////////////////////////////
    # /////////////////////////////////////////////////     Вывод среднего значения по атрибуту(Стоимость тарифа, лимит Гб и тд.)
    def average(self, attr):
        num = 0
        sum = 0
        for i in self.dict:
            for key, val in self.dict[i].items():
                if key == attr:
                    sum += float(val)
                    num += 1
        if num != 0:
            average = sum / num
            print(f"Средний показатель по атрибуту:\n{attr}:\t{average}")
    # /////////////////////////////////////////////////
    
close = True
command = ""
mobile = Mobile_operator()
while close == True:
    instruction = input("Введите запрос: ")
    if "Add" in instruction:
        command = instruction.split(" ")
        mobile.name = command[1]
        mobile.call_cost = command[2]
        mobile.gb_limit = command[3]
        mobile.sms_limit = command[4]
        mobile.min_limit = command[5]
        mobile.add()
        command = ""
    if "Delete" in instruction:
        command = instruction.split(" ")
        mobile.name = command[1]
        mobile.delete()
        command = ""
    if "Search" in instruction:
        command = instruction.split(" ")
        mobile.name = command[1]
        mobile.search()
        command = ""
    if "Filter" in instruction:
        command = instruction.split(" ")
        mobile.filter(command[1], command[2], command[3])
        command = ""
    if "Average_stat" in instruction:
        command = instruction.split(" ")
        mobile.average(command[1])
        command = ""
    if instruction == "List_obj":
        mobile.list_obj()
    if instruction == "Close":
        close = False