"""
Script para limpar todas as tabelas do banco de dados
"""
from sqlalchemy import text
from src.database.config import SessionLocal, engine

def clear_all_tables():
    """Limpa todas as tabelas do banco de dados"""
    session = SessionLocal()
    try:
        # Lista de tabelas para limpar (na ordem correta por causa das FKs)
        tables = [
            "performance_logs",
            "alerts", 
            "results",
            "files"
        ]
        
        print("[INFO] Limpando banco de dados...")
        
        for table in tables:
            result = session.execute(text(f"DELETE FROM {table}"))
            count = result.rowcount
            print(f"   [OK] Tabela '{table}': {count} registros removidos")
        
        session.commit()
        print("\n[OK] Banco de dados limpo com sucesso")
        
    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] Erro ao limpar banco: {e}")
        raise
    finally:
        session.close()

def drop_all_tables():
    """Remove todas as tabelas (estrutura completa)"""
    with engine.begin() as conn:
        try:
            print("[INFO] Removendo todas as tabelas...")
            
            # Remove todas as tabelas
            conn.execute(text("DROP TABLE IF EXISTS performance_logs CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS alerts CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS results CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS files CASCADE;"))
            
            print("[OK] Todas as tabelas foram removidas")
            print("[WARN] Execute a API para recriar as tabelas")
            
        except Exception as e:
            print(f"[ERROR] Erro ao remover tabelas: {e}")
            raise

if __name__ == "__main__":
    print("=" * 60)
    print("AutoMLOps - Limpeza de Banco de Dados")
    print("=" * 60)
    print()
    print("Escolha uma opção:")
    print("1 - Limpar dados (mantém estrutura das tabelas)")
    print("2 - Remover TUDO (apaga estrutura e dados)")
    print("0 - Cancelar")
    print()
    
    choice = input("Digite sua escolha: ").strip()
    
    if choice == "1":
        print()
        confirm = input("[?] Confirma limpar TODOS os dados? (sim/nao): ").strip().lower()
        if confirm == "sim":
            clear_all_tables()
        else:
            print("[INFO] Operação cancelada")
    
    elif choice == "2":
        print()
        confirm = input("[?] Confirma REMOVER TODAS AS TABELAS? (sim/nao): ").strip().lower()
        if confirm == "sim":
            drop_all_tables()
        else:
            print("[INFO] Operação cancelada")
    
    else:
        print("[INFO] Operação cancelada")
